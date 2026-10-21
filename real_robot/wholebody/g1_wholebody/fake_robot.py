"""A simulated G1 behind the real robot's interfaces (development only; needs MuJoCo).

* takes LowCmd exactly as the firmware does: CRC checked (a bad CRC is ignored), float32 values, the motor PD law
  tau = tau_ff + kp (q* - q) + kd (dq* - dq) clipped to the torque limits, executed at 500 Hz; motors keep the
  last command they received; motors with mode 0 produce no torque
* reports LowState: joints, actuator torques, pelvis IMU (quaternion with an arbitrary yaw offset, gyro, acc),
  tick, wireless remote
* reports a SLAM-like pose of the Mid-360 IMU (T_map_body) through a chosen 'true' LiDAR mount
* optional gantry rope on the pelvis (elastic when stretched) to rehearse stand-up and hand-over
Virtual scene boxes are drawn translucent and never collide unless physical=True.
"""
import re
from pathlib import Path

import numpy as np

from wholebody_core import WHOLEBODY, quat_matrix
from .frames import PelvisChain, pose_matrix, yaw_matrix, matrix_quat
from .lowlevel import LowCmd, LowState, NUM_BODY, crc_of_packed, remote_bytes


class FakeG1:
    def __init__(self, contract, kinematics, scene=None, physical=False, start=(0., 0., 0.), assets=None,
                 mount='unitree_urdf', motor_kp_scale=1.0, imu_yaw_offset=0.7, gantry=None, seed=0):
        import mujoco
        self.mj = mujoco
        self.c = contract
        xml = (WHOLEBODY / 'robot.xml').read_text()
        if assets:
            xml = xml.replace('file="assets/', 'file="%s/' % Path(assets).resolve())
        else:                                  # no meshes: visual geoms only, not needed for physics
            xml = re.sub(r'<mesh [^>]*/>', '', xml); xml = re.sub(r'<geom [^>]*mesh="[^"]*"[^>]*/>', '', xml)
        geoms = []
        for k, b in enumerate(scene.boxes if scene is not None else []):
            geoms.append('<geom name="obstacle_%d" type="box" pos="%s" size="%s" euler="0 0 %.6f" contype="0" conaffinity="%d" '
                         'rgba="0.9 0.4 0.2 0.35"/>' % (k, ' '.join('%.5f' % v for v in b['center']),
                                                        ' '.join('%.5f' % v for v in b['half_size']), b['yaw'], 2 if physical else 0))
        xml = re.sub(r'(<worldbody>)', r'\1' + ''.join(geoms), xml, count=1)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.timestep = contract['physics_dt']
        if physical:
            robot = self.model.geom_bodyid > 0
            self.model.geom_contype[robot & (self.model.geom_contype == 0)] = 2
        self.data = mujoco.MjData(self.model)
        actuated = [self.model.actuator(i).trnid[0] for i in range(self.model.nu)]
        if [self.model.joint(j).name for j in actuated] != list(contract['joint_order']):
            raise ValueError('actuator order differs from the policy joint order')
        self.gyro_adr = self.model.sensor_adr[self.model.sensor('gyro_pelvis').id]
        self.acc_adr = self.model.sensor_adr[self.model.sensor('accelerometer_pelvis').id]
        self.pelvis_id = self.model.body('pelvis').id
        self.chain = PelvisChain(kinematics, mount)
        self.motor_kp_scale = motor_kp_scale
        self.imu_yaw = yaw_matrix(imu_yaw_offset)
        self.rng = np.random.default_rng(seed)
        self.gantry = gantry                    # dict(anchor=[x, y, z], length=L, k=..., d=...) or None
        self.cmd = None; self.queue = []        # (apply_time, arrays)
        self.bad_crc = 0; self.tick = 0; self.time = 0.; self.buttons = (); self.sticks = (0., 0., 0., 0.)
        x, y, yaw = start
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [x, y, .8]; self.data.qpos[3:7] = matrix_quat(yaw_matrix(yaw))
        self.data.qpos[7:] = contract['nominal']
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ robot interface
    def send(self, cmd: LowCmd, apply_at):
        if crc_of_packed(cmd.pack(crc=0)) != cmd.crc:
            self.bad_crc += 1; return False
        arrays = {k: cmd.as_float32(k)[:NUM_BODY] for k in ('q', 'dq', 'tau', 'kp', 'kd')}
        arrays['on'] = cmd.mode[:NUM_BODY] == 1
        self.queue.append((apply_at, arrays))
        return True

    def lowstate(self):
        d = self.data
        quat = matrix_quat(self.imu_yaw @ quat_matrix(d.qpos[3:7]))       # IMU yaw is arbitrary on the robot
        return LowState(tick=self.tick, mode_pr=0, mode_machine=5, q=d.qpos[7:].copy(), dq=d.qvel[6:].copy(),
                        tau=d.actuator_force.copy(), motor_mode=np.ones(NUM_BODY, np.uint8), temperature=np.full(NUM_BODY, 40),
                        quat_wxyz=quat, gyro=d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy(),
                        acc=d.sensordata[self.acc_adr:self.acc_adr + 3].copy(), rpy=np.zeros(3),
                        remote=remote_bytes(*self.buttons, sticks=self.sticks), received=None)

    def slam_pose(self, noise_xy=0., noise_yaw=0.):
        """T_map_body of the Mid-360 IMU now (truth through the TRUE mount), with optional noise."""
        t = self.chain.body_from_pelvis(pose_matrix(self.data.qpos[:3], self.data.qpos[3:7]), self.data.qpos[7:].copy())
        if noise_xy or noise_yaw:
            t = t.copy(); t[:2, 3] += self.rng.normal(0, noise_xy, 2)
            t[:3, :3] = yaw_matrix(self.rng.normal(0, noise_yaw)) @ t[:3, :3]
        return t

    # ------------------------------------------------------------------ physics
    def step(self):
        mj, d, c = self.mj, self.data, self.c
        while self.queue and self.queue[0][0] <= self.time + 1e-9:
            self.cmd = self.queue.pop(0)[1]
        torque = np.zeros(NUM_BODY)
        if self.cmd is not None:
            q, dq = d.qpos[7:], d.qvel[6:]
            cm = self.cmd
            torque = cm['tau'] + self.motor_kp_scale * cm['kp'] * (cm['q'] - q) + cm['kd'] * (cm['dq'] - dq)
            torque = np.where(cm['on'], np.clip(torque, -c['torque_limit'], c['torque_limit']), 0.)
        d.ctrl[:] = torque
        d.qfrc_applied[:] = 0.
        if self.gantry is not None:
            # rope hooked on the back of the torso, 0.25 m above torso_link (as the gantry harness): a pendulum
            torso = self.model.body('torso_link').id
            hook = d.xpos[torso] + d.xmat[torso].reshape(3, 3) @ np.array([-0.08, 0., 0.25])
            anchor = np.asarray(self.gantry['anchor'], float)
            rope = anchor - hook; dist = np.linalg.norm(rope); stretch = dist - self.gantry['length']
            if stretch > 0:
                u = rope / dist
                vel = np.zeros(6); mj.mj_objectVelocity(self.model, d, mj.mjtObj.mjOBJ_BODY, torso, vel, 0)
                force = (self.gantry.get('k', 3000.) * stretch - self.gantry.get('d', 300.) * min(0., vel[3:] @ u)) * u
                mj.mj_applyFT(self.model, d, force, np.zeros(3), hook, torso, d.qfrc_applied)
        mj.mj_step(self.model, d)
        self.time += c['physics_dt']; self.tick += 1

    # ------------------------------------------------------------------ truth for tests
    @property
    def pelvis(self):
        return self.data.qpos[:3].copy(), self.data.qpos[3:7].copy()

    def foot_load(self):
        """Fraction of the robot's weight carried by the floor (0 hanging, ~1 standing)."""
        total = 0.
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if self.model.geom_bodyid[con.geom1] == 0 or self.model.geom_bodyid[con.geom2] == 0:
                f = np.zeros(6); self.mj.mj_contactForce(self.model, self.data, i, f); total += abs(f[0])
        return total / (self.model.body_subtreemass[1] * 9.81)
