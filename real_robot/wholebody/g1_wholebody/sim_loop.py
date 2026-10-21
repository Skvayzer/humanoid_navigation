#!/usr/bin/env python3
"""Closed loop on simulated time: the robot executor (runner.Executor) against FakeG1 (MuJoCo).

Everything between the robot interfaces is the code that runs on the G1: pelvis chain, IMU tilt fusion, field
planner, observation builder, ONNX policy, LowCmd + CRC, watchdogs. Unlike cat/wholebody_sim.py the SLAM pose
really updates at 10 Hz (held in between, plus latency and noise), commands reach the motors late, fields arrive
seconds late (Jetson), and the run starts the way it will on the robot: hanging on the gantry, stand-up, lower,
balance, release, go.

  python -m g1_wholebody.sim_loop --scene sidegap_corridor_048 --policy robust_delay_v1_u150 [--video out.mp4]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from wholebody_core import WHOLEBODY, load_contract
from .kinematics import FastKinematics
from . import runner as rn
from .fake_robot import FakeG1
from .fields import FieldPlanner, VirtualScene, solve_fields
from .frames import heading, yaw_matrix

HANG_LENGTH = 2.2 - (0.79 + 0.03 + 0.42)   # anchor height - (standing pelvis + 3 cm clearance + pelvis->hook)
POLICIES = {'u596': WHOLEBODY, 'robust_delay_v1_u150': WHOLEBODY / 'robust_delay_v1_u150'}


class SimClock:
    def __init__(self):
        self.t = 0.

    def __call__(self):
        return self.t


class DelayedFields:
    """Field source with the Jetson's solve time on the simulated clock. request() only queues (as the real worker
    does: the control step never pays for the solve); service() solves outside the timed step; poll() delivers
    the result field_latency after the request."""

    def __init__(self, scene, latency_s, clock):
        self.scene, self.latency, self.clock = scene, latency_s, clock
        self.pending = None; self.ready = None; self.solves = 0

    def request(self, base_xy, goal_xy):
        if self.pending is not None or self.ready is not None:
            return False
        self.pending = (self.clock(), np.array(base_xy, float), np.array(goal_xy, float))
        return True

    def service(self):
        if self.pending is not None:
            t, xy, goal = self.pending; self.pending = None
            fields, used = solve_fields(xy, goal, self.scene.occupancy)
            self.ready = (t + self.latency, fields, used); self.solves += 1

    def poll(self):
        if self.ready is not None and self.clock() >= self.ready[0]:
            _, fields, goal = self.ready; self.ready = None
            return fields, goal
        return None


def make_session(policy):
    import onnxruntime as ort
    path = POLICIES.get(policy, Path(policy)) / 'policy.onnx'
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    return lambda obs: session.run(None, {'obs': obs[None].astype(np.float32)})[0][0]


def gantry_schedule(t_go=11.0):
    """(time, kind, value): the hand-over procedure of the robot checklist."""
    return [(1.0, 'cmd', 'stand'), (5.0, 'rope', 0.05), (7.0, 'cmd', 'balance'), (8.0, 'rope', 9.0), (t_go, 'cmd', 'go')]


def run(scene='sidegap_corridor_048', policy='robust_delay_v1_u150', anchor=(1.0, 0.5, 0.4), seconds=40.,
        procedure='gantry', pose_hz=10., pose_latency_s=0.06, pose_noise=0.01, action_latency_s=0.010,
        field_latency_s=2.5, field_period_s=1.0, motor_kp_scale=0.9, true_mount='unitree_urdf', exec_mount='unitree_urdf',
        events=(), seed=0, video=None, assets=None, fence_margin=1.0, physical=False, imu_heading=True):
    contract = load_contract(); kin = FastKinematics(contract)
    clock = SimClock()
    scene_obj = VirtualScene(scene, anchor=anchor)
    planner = FieldPlanner(scene_obj, period=field_period_s)
    fields = DelayedFields(scene_obj, field_latency_s, clock)
    xs = [scene_obj.start[0], scene_obj.goal[0]] + [b['center'][0] for b in scene_obj.boxes]
    ys = [scene_obj.start[1], scene_obj.goal[1]] + [b['center'][1] for b in scene_obj.boxes]
    fence = (min(xs) - fence_margin, max(xs) + fence_margin, min(ys) - fence_margin, max(ys) + fence_margin)
    cfg = rn.Config(mount=exec_mount, allow_active=True, geofence=fence, imu_heading=imu_heading)
    ex = rn.Executor(contract, kin, make_session(policy), planner, fields, cfg, clock=clock)
    hang = None
    if procedure == 'gantry':
        x, y = scene_obj.start
        hang = dict(anchor=[x - 0.08, y, 2.2], length=HANG_LENGTH, k=3000., d=300.)
    robot = FakeG1(contract, kin, scene_obj, physical=physical, start=(*scene_obj.start, anchor[2]), mount=true_mount,
                   motor_kp_scale=motor_kp_scale, gantry=hang, seed=seed, assets=assets)
    schedule = sorted((gantry_schedule() if procedure == 'gantry' else []) + list(events))
    pose_queue = []; next_pose = 0.; steps = int(seconds / contract['control_dt'])
    renderer = frames = None
    if video:
        import mujoco
        renderer = mujoco.Renderer(robot.model, 480, 640); frames = []
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = 1
        cam.distance, cam.elevation, cam.azimuth = 3.2, -30, 150
    log = dict(modes=[], intrusions=0, intrusion_sites={}, min_z=9., max_step_ms=0., latency_ms=[], transitions=[],
               fell=False)
    last_mode = ex.mode
    for k in range(steps):
        now = k * contract['control_dt']
        while schedule and schedule[0][0] <= now:
            _, kind, value = schedule.pop(0)
            if kind == 'cmd':
                ok = ex.command(value); log['transitions'].append((round(now, 2), 'cmd ' + value, ok, ex.reason))
            elif kind == 'rope' and robot.gantry is not None:
                robot.gantry['length'] = HANG_LENGTH + value if value < 1 else 10.
            elif kind == 'buttons':
                robot.buttons = tuple(value)
            elif kind == 'drop_lowstate':
                log['drop_lowstate_until'] = now + value
            elif kind == 'freeze_pose':
                log['freeze_pose_until'] = now + value
            elif kind == 'pose_jump':
                log['pose_jump'] = value
        clock.t = now
        # SLAM: a pose of the current instant every 1/pose_hz, delivered pose_latency later
        if now >= next_pose - 1e-9:
            if now >= log.get('freeze_pose_until', -1):
                t = robot.slam_pose(pose_noise, pose_noise)
                if 'pose_jump' in log:
                    t = t.copy(); t[0, 3] += log.pop('pose_jump')
                pose_queue.append((now + pose_latency_s, t, now))
            next_pose = now + 1. / pose_hz
        while pose_queue and pose_queue[0][0] <= now + 1e-9:
            _, t, stamp = pose_queue.pop(0); ex.on_slam_pose(t, stamp, received=now, age_s=now - stamp)
        if now >= log.get('drop_lowstate_until', -1):
            state = robot.lowstate(); state.received = now - 0.001; ex.on_lowstate(state)
        if procedure == 'policy' and ex.mode == rn.IDLE and ex.fields is not None and ex.pose is not None:
            ex.mode = rn.POLICY; ex.policy_started = now; ex.mode_since = now    # test hook: walk at once, like the sim
            log['transitions'].append((round(now, 2), 'IDLE->POLICY (test hook)', True, ''))
            last_mode = ex.mode
        cmd = ex.step(now)
        fields.service()
        log['max_step_ms'] = max(log['max_step_ms'], 1e3 * ex.last['step_s'])
        if cmd is not None:
            robot.send(cmd, apply_at=robot.time + action_latency_s)
        if ex.mode != last_mode:
            log['transitions'].append((round(now, 2), last_mode + '->' + ex.mode, True, ex.reason)); last_mode = ex.mode
        frozen = procedure == 'policy' and ex.mode == rn.IDLE        # sim-like start: standing still until the policy runs
        for _ in range(contract['substeps']):
            if frozen:
                robot.time += contract['physics_dt']; continue
            robot.step()
        pos, quat = robot.pelvis
        log['min_z'] = min(log['min_z'], float(pos[2]))
        if ex.mode == rn.POLICY:
            sites = kin.world_sites(robot.data.qpos[7:].copy(), pos, quat)
            inside = scene_obj.inside_any_box(sites)
            if inside.any():
                log['intrusions'] += 1
                for i in np.flatnonzero(inside):
                    name = kin.site_names[i]; log['intrusion_sites'][name] = log['intrusion_sites'].get(name, 0) + 1
            if pos[2] < .45 and (robot.gantry is None or robot.gantry['length'] > 5):
                log['fell'] = True
            if ex.last.get('policy_wait'):
                log['wait_steps'] = log.get('wait_steps', 0) + 1
        if ex.mode == rn.POLICY and planner.hold_xy is not None:     # 'balance' / 'stop': the robot must stay put
            log.setdefault('hold_from', pos[:2].copy())
            log['hold_drift'] = max(log.get('hold_drift', 0.), float(np.linalg.norm(pos[:2] - log['hold_from'])))
        else:
            log.pop('hold_from', None)
        if renderer is not None and k % 2 == 0:
            renderer.update_scene(robot.data, cam); frames.append(renderer.render().copy())
        if planner.reached and ex.mode == rn.POLICY and np.linalg.norm(robot.data.qvel[:2]) < .05 and now > 1.:
            break
        if log['fell']:
            break
    pos, _ = robot.pelvis
    result = dict(scene=scene, policy=policy, procedure=procedure, seconds=round(now, 2), reached=bool(planner.reached),
                  fell=log['fell'], final_mode=ex.mode, final_reason=ex.reason, final_xy=pos[:2].round(3).tolist(),
                  goal_xy=np.round(scene_obj.goal, 3).tolist(), site_intrusion_steps=log['intrusions'],
                  intrusion_sites=log['intrusion_sites'], min_pelvis_z=round(log['min_z'], 3),
                  max_step_ms=round(log['max_step_ms'], 2), field_solves=fields.solves, bad_crc=robot.bad_crc,
                  hold_drift_m=round(log.get('hold_drift', 0.), 3),
                  policy_wait_s=round(log.get('wait_steps', 0) * contract['control_dt'], 2),
                  transitions=log['transitions'],
                  settings=dict(pose_hz=pose_hz, pose_latency_s=pose_latency_s, pose_noise=pose_noise, action_latency_s=action_latency_s,
                                field_latency_s=field_latency_s, motor_kp_scale=motor_kp_scale, true_mount=true_mount,
                                exec_mount=exec_mount, anchor=list(anchor), seed=seed))
    if renderer is not None and frames:
        from wholebody_sim import write_video
        write_video(video, frames, fps=25); result['video'] = str(video)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', default='sidegap_corridor_048'); p.add_argument('--policy', default='robust_delay_v1_u150')
    p.add_argument('--procedure', choices=('gantry', 'policy'), default='gantry')
    p.add_argument('--anchor', type=float, nargs=3, default=(1.0, 0.5, 0.4), metavar=('X', 'Y', 'YAW'))
    p.add_argument('--seconds', type=float, default=40.); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pose-hz', type=float, default=10.); p.add_argument('--pose-latency', type=float, default=0.06)
    p.add_argument('--pose-noise', type=float, default=0.01); p.add_argument('--action-latency', type=float, default=0.010)
    p.add_argument('--field-latency', type=float, default=2.5); p.add_argument('--motor-kp-scale', type=float, default=0.9)
    p.add_argument('--true-mount', default='unitree_urdf'); p.add_argument('--exec-mount', default='unitree_urdf')
    p.add_argument('--physical', action='store_true'); p.add_argument('--video'); p.add_argument('--assets'); p.add_argument('--out')
    a = p.parse_args()
    t = time.time()
    r = run(a.scene, a.policy, tuple(a.anchor), a.seconds, a.procedure, a.pose_hz, a.pose_latency, a.pose_noise,
            a.action_latency, a.field_latency, motor_kp_scale=a.motor_kp_scale, true_mount=a.true_mount,
            exec_mount=a.exec_mount, seed=a.seed, video=a.video, assets=a.assets, physical=a.physical)
    r['wall_s'] = round(time.time() - t, 1)
    text = json.dumps(r, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
