#!/usr/bin/env python3
"""Sim-to-sim: the robot-side whole-body stack, closed loop, against plain MuJoCo physics.

Development tool only (needs MuJoCo + ONNX Runtime; never part of the robot image). What runs is exactly what the
robot would run: wholebody_fields (occupancy -> fields in a rolling 5.12 m window, recomputed every second toward
the active waypoint), wholebody_core.ObservationBuilder (226 inputs from joint state, IMU, pelvis pose, fields),
the exported ONNX policy, and the actuation law (kp/kd PD on the targets + waist/arm gravity feedforward) at the
physics rate. Unlike training, obstacles are PHYSICAL here (robot links collide with them) and every robot-obstacle
contact is counted.

  python cat/wholebody_sim.py --bundle <deploy bundle dir> --scene course_mixed_25m [--video out.mp4]
      [--pose-noise 0.01] [--pose-latency 2] [--action-delay 1] [--kp-scale 0.9] [--robot-perception]
"""
import argparse
import json
import math
import re
import time
from collections import deque
from pathlib import Path

import numpy as np

from wholebody_core import ObservationBuilder, WholeBodyKinematics, WHOLEBODY, load_contract, pd_torque, quat_matrix
from wholebody_fields import training_fields

DX = .04
WINDOW = (128, 128, 38)        # 5.12 x 5.12 x 1.52 m, voxel centers from z = 0.02 (training grids)


def box_corners_test(boxes, centers, dx):
    """Conservative occupancy: a cell is occupied when the cube [c - dx/2, c + dx/2] overlaps a (yawed) box."""
    occupied = np.zeros(len(centers), bool)
    h = dx/2
    for b in boxes:
        c, half, yaw = np.asarray(b['center'], float), np.asarray(b['half_size'], float), float(b.get('yaw', 0.))
        ca, sa = math.cos(yaw), math.sin(yaw)
        d = centers-c
        # separating-axis test in xy (box axes and world axes) + z interval
        lx, ly = d[:, 0]*ca+d[:, 1]*sa, -d[:, 0]*sa+d[:, 1]*ca
        ex = h*(abs(ca)+abs(sa))
        box_x = half[0]*abs(ca)+half[1]*abs(sa); box_y = half[0]*abs(sa)+half[1]*abs(ca)
        overlap = (np.abs(lx) < half[0]+ex) & (np.abs(ly) < half[1]+ex)
        overlap &= (np.abs(d[:, 0]) < box_x+h) & (np.abs(d[:, 1]) < box_y+h) & (np.abs(d[:, 2]) < half[2]+h)
        occupied |= overlap
    return occupied


class Scene:
    def __init__(self, name):
        path = Path(name)
        self.spec = json.loads((path if path.suffix == '.json' else WHOLEBODY/'scenes'/(name+'.json')).read_text())
        self.boxes = self.spec['boxes']
        self.start = np.asarray(self.spec['start'][:2], float)
        self.goal = np.asarray(self.spec['goal'][:2], float)

    def occupancy(self, first_center, shape, drop_below=0., ceiling=None):
        axes = [first_center[i]+np.arange(shape[i])*DX for i in range(3)]
        centers = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        occ = box_corners_test(self.boxes, centers, DX).reshape(shape)
        z = axes[2]
        if drop_below > 0:
            occ[:, :, z < drop_below] = False          # robot perception: returns below 10 cm are ground
        if ceiling is not None:
            occ[:, :, z > ceiling] = False             # robot perception volume ends at 1.4 m
        return occ

    def waypoints(self, spacing=1.6):
        """Goals for the rolling field window: along the start->goal line every `spacing` m (navigation layer)."""
        d = self.goal-self.start; n = max(1, int(math.ceil(np.linalg.norm(d)/spacing)))
        return [self.start+d*(k/n) for k in range(1, n+1)]


def build_model(bundle, scene, physical=True):
    import mujoco
    xml = (Path(bundle)/'robot.xml').read_text()
    xml = xml.replace('file="assets/', 'file="%s/assets/' % Path(bundle).resolve())
    geoms = []
    for k, b in enumerate(scene.boxes):
        size = ' '.join('%.5f' % v for v in b['half_size']); pos = ' '.join('%.5f' % v for v in b['center'])
        geoms.append('<geom name="obstacle_%d" type="box" pos="%s" size="%s" euler="0 0 %.6f" contype="0" conaffinity="%d" '
                     'rgba="0.62 0.66 0.70 %s"/>' % (k, pos, size, float(b.get('yaw', 0.)), 2 if physical else 0,
                                                       '0.18' if 'wall' in b.get('category', '') else '1'))
    xml = re.sub(r'(<worldbody>)', r'\1'+''.join(geoms), xml, count=1)
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = .002
    obstacle = np.array([model.geom(i).name.startswith('obstacle_') for i in range(model.ngeom)])
    robot = model.geom_bodyid > 0
    if physical:
        # Robot links collide with obstacles only (contype 2 vs conaffinity 2): no new self/floor contacts.
        model.geom_contype[robot & (model.geom_contype == 0)] = 2
    return model, obstacle, robot


def run(args):
    import mujoco
    import onnxruntime as ort
    contract = load_contract(); kin = WholeBodyKinematics(contract)
    scene = Scene(args.scene)
    model, obstacle_geom, robot_geom = build_model(args.bundle, scene, physical=not args.ghost)
    data = mujoco.MjData(model)
    actuated = [model.actuator(i).trnid[0] for i in range(model.nu)]
    if [model.joint(j).name for j in actuated] != contract['joint_order']:
        raise ValueError("actuator order differs from the policy joint order")
    gyro_adr = model.sensor_adr[model.sensor('gyro_pelvis').id]
    session = ort.InferenceSession(str(Path(args.bundle)/'policy.onnx'), providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(args.seed)
    kp_scale = args.kp_scale

    # ------------------------------------------------------------ reset (standing at the start, facing +x)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [scene.start[0], scene.start[1], .8]; data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = contract['nominal']
    mujoco.mj_forward(model, data)
    waypoints = scene.waypoints(args.waypoint_spacing); active = 0
    builder = ObservationBuilder(contract, kin, frequency=args.frequency)

    held = {'step': -1, 'noise': (np.zeros(2), 0.)}
    def pose():
        lag = poses[0] if args.pose_latency else (data.qpos[:3].copy(), data.qpos[3:7].copy())
        if args.pose_noise and args.pose_period > 1:
            # Localisation updates at a fixed rate: the error of the last update is held in between (SLAM ~10 Hz).
            if held['step'] < 0 or len(poses) and stepper['n']-held['step'] >= args.pose_period:
                held['step'] = stepper['n']; held['noise'] = (rng.normal(0, args.pose_noise, 2), rng.normal(0, args.pose_noise))
            pos, quat = lag[0].copy(), lag[1].copy(); pos[:2] += held['noise'][0]
            yaw = held['noise'][1]
            return pos, quat_mul(quat, np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)]))
        pos, quat = lag[0].copy(), lag[1].copy()
        if args.pose_noise:
            pos[:2] += rng.normal(0, args.pose_noise, 2)
            yaw = rng.normal(0, args.pose_noise)          # ~ radians per metre of position noise
            quat = quat_mul(quat, np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)]))
        return pos, quat

    stepper = {'n': 0}
    poses = deque(maxlen=max(1, args.pose_latency+1))
    poses.append((data.qpos[:3].copy(), data.qpos[3:7].copy()))
    fields, field_center, field_goal = None, None, None
    def update_fields(base_xy):
        nonlocal fields, field_center, field_goal
        center = np.round(base_xy/DX)*DX
        first = np.array([center[0]-WINDOW[0]/2*DX+DX/2, center[1]-WINDOW[1]/2*DX+DX/2, DX/2])
        goal = np.r_[waypoints[active], .75]
        lo, hi = first[:2]+.3, first[:2]+(np.array(WINDOW[:2])-1)*DX-.3
        goal[:2] = np.clip(goal[:2], lo, hi)            # waypoint beyond the window: its projection on the edge
        occ = scene.occupancy(first, WINDOW, drop_below=.10 if args.robot_perception else 0.,
                              ceiling=1.4 if args.robot_perception else None)
        free = ~occ[..., int(round((.75-DX/2)/DX))]
        if not free[tuple(np.clip(np.round((goal[:2]-first[:2])/DX).astype(int), 0, np.array(WINDOW[:2])-1))]:
            idx = np.argwhere(free); target = np.round((goal[:2]-first[:2])/DX)
            goal[:2] = first[:2]+idx[np.argmin(((idx-target)**2).sum(-1))]*DX
        fields = training_fields(occ, first, DX, goal); field_center, field_goal = center, goal
    update_fields(data.qpos[:2].copy())
    gyro = data.sensordata[gyro_adr:gyro_adr+3].copy()
    obs = builder.reset(data.qpos[7:].copy(), *pose(), fields, dq=data.qvel[6:].copy(), gyro=gyro)

    renderer = None
    if args.video:
        renderer = mujoco.Renderer(model, 480, 640); frames = []
        camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING; camera.trackbodyid = 1
        camera.distance, camera.elevation, camera.azimuth = 3.0, -35, 150
    actions = deque([np.zeros(29)]*(args.action_delay+1), maxlen=args.action_delay+1)
    issued = deque([builder.targets.copy()]*3, maxlen=3)     # targets sent in the last three control periods
    contact_steps, contact_parts, fell, reached = 0, {}, False, False
    trace = {k: [] for k in ('base', 'quat', 'vel', 'command', 'command_delay', 'sdf', 'action', 'active')}
    timing = dict(fields=0., observe=0., policy=0., physics=0.)
    steps = int(args.seconds/contract['control_dt'])
    for step in range(steps):
        t = time.time()
        if step % int(args.field_period/contract['control_dt']) == 0 and step:
            update_fields(data.qpos[:2].copy())
        timing['fields'] += time.time()-t; t = time.time()
        action = session.run(None, {'obs': obs[None].astype(np.float32)})[0][0]
        timing['policy'] += time.time()-t
        actions.append(action)
        previous_targets = builder.targets.copy()
        targets = builder.act(actions[0])
        issued.append(targets.copy())
        late = int(round(args.action_latency_ms/1000/contract['physics_dt']))
        t = time.time()
        feedforward = np.zeros(29)
        if contract['upper_gravity_compensation']:
            feedforward[12:] = kin.upper_gravity(data.qpos[7:], data.qpos[3:7])
        touching = False
        for substep in range(contract['substeps']):
            q, dq = data.qpos[7:], data.qvel[6:]
            # Motor-side latency (as training and the robot: the runtime knows what it commanded, the motors act on
            # it late): track the newest targets issued at least `late` physics substeps ago (up to 2 control periods).
            age = 0 if substep >= late else (1 if substep+contract['substeps'] >= late else 2)
            applied = issued[-1-age]
            torque = kp_scale*contract['kp']*(applied-q)-contract['kd']*dq+feedforward
            data.ctrl[:] = np.clip(torque, -contract['torque_limit'], contract['torque_limit'])
            mujoco.mj_step(model, data)
            for c in data.contact[:data.ncon]:
                if (obstacle_geom[c.geom1] and robot_geom[c.geom2]) or (obstacle_geom[c.geom2] and robot_geom[c.geom1]):
                    touching = True
                    body = model.body(model.geom_bodyid[c.geom2 if obstacle_geom[c.geom1] else c.geom1]).name
                    contact_parts[body] = contact_parts.get(body, 0)+1
        timing['physics'] += time.time()-t
        contact_steps += touching
        poses.append((data.qpos[:3].copy(), data.qpos[3:7].copy()))
        stepper['n'] += 1
        t = time.time()
        gyro = data.sensordata[gyro_adr:gyro_adr+3].copy()
        obs = builder.observe(data.qpos[7:].copy(), data.qvel[6:].copy(), gyro, *pose(), fields)
        timing['observe'] += time.time()-t
        if args.trace:
            for k, v in (('base', data.qpos[:3]), ('quat', data.qpos[3:7]), ('vel', data.qvel[:6]), ('command', builder.command),
                         ('command_delay', builder.command_delay), ('sdf', builder.sdf_delay), ('action', action), ('active', [active])):
                trace[k].append(np.array(v, dtype=float).copy())
        xy = data.qpos[:2]
        if active < len(waypoints)-1 and np.linalg.norm(waypoints[active]-xy) < args.advance_radius:
            active += 1; update_fields(xy.copy())
            if builder.stop <= 0:
                builder.rearm()
        if active == len(waypoints)-1 and np.linalg.norm(waypoints[-1]-xy) < .4:
            reached = True
        if data.qpos[2] < .45:
            fell = True
            break
        if renderer is not None and step % 2 == 0:
            renderer.update_scene(data, camera); frames.append(renderer.render().copy())
        if reached and step*contract['control_dt'] > 1. and np.linalg.norm(data.qvel[:2]) < .05:
            break
    result = dict(scene=scene.spec['scene_id'], seconds=round((step+1)*contract['control_dt'], 2), reached_goal=bool(reached),
                  fell=bool(fell), final_xy=[round(float(v), 3) for v in data.qpos[:2]], goal_xy=scene.goal.tolist(),
                  progress_m=round(float(data.qpos[0]-scene.start[0]), 3), contact_steps=int(contact_steps),
                  contact_bodies=dict(sorted(contact_parts.items(), key=lambda kv: -kv[1])[:6]),
                  settings=dict(pose_noise=args.pose_noise, pose_period=args.pose_period, pose_latency=args.pose_latency, action_delay=args.action_delay,
                                action_latency_ms=args.action_latency_ms, kp_scale=kp_scale, robot_perception=args.robot_perception, ghost=args.ghost, frequency=args.frequency),
                  ms_per_step={k: round(1000*v/(step+1), 2) for k, v in timing.items()})
    if args.trace:
        np.savez_compressed(args.trace, **{k: np.asarray(v) for k, v in trace.items()}, waypoints=np.asarray(waypoints))
    if renderer is not None and frames:
        write_video(args.video, frames, fps=25)
        result['video'] = str(args.video)
    return result


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def write_video(path, frames, fps):
    import subprocess
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (w, h),
                             '-r', str(fps), '-i', '-', '-pix_fmt', 'yuv420p', '-vcodec', 'libx264', str(path)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close(); proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', required=True, help='deploy bundle directory (policy.onnx, robot.xml, assets/)')
    p.add_argument('--scene', default='course_mixed_25m')
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--field-period', type=float, default=1., help='field recomputation period (s)')
    p.add_argument('--waypoint-spacing', type=float, default=1.6)
    p.add_argument('--advance-radius', type=float, default=.8)
    p.add_argument('--frequency', type=float, default=1.4, help='gait frequency (training: U(1.3, 1.5) Hz)')
    p.add_argument('--pose-noise', type=float, default=0., help='pelvis xy noise std (m) and yaw noise std (rad)')
    p.add_argument('--pose-latency', type=int, default=0, help='pose latency in control steps (20 ms)')
    p.add_argument('--pose-period', type=int, default=1, help='pose noise refreshed every N control steps (SLAM rate); 1 = every step')
    p.add_argument('--action-delay', type=int, default=0, help='action latency in control steps (20 ms)')
    p.add_argument('--action-latency-ms', type=float, default=0., help='observation-to-motor latency within a step (2 ms resolution)')
    p.add_argument('--kp-scale', type=float, default=1.)
    p.add_argument('--robot-perception', action='store_true', help='drop obstacles below 10 cm and above 1.4 m (robot volume)')
    p.add_argument('--ghost', action='store_true', help='non-physical obstacles (as in training); contacts still counted')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--video')
    p.add_argument('--trace', help='npz with per-step base pose, velocity, commands, field distances, actions')
    p.add_argument('--out')
    args = p.parse_args()
    result = run(args)
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1)+'\n')


if __name__ == '__main__':
    main()
