"""Whole-body CAT policy runtime (numpy only): observations, actions, kinematics, gravity feedforward.

Policy: generalist_finetune_v1_u596 (Click-and-Traverse-Mjlab), 226 observations -> 29 joint actions.
Every function here is a port of the training code path this policy was trained with in FIELD-DRIVEN (goal) mode
(cat_mjlab/task.py step/_fields/_elbow_fields/_observe/reset, task_math.py). Constants come from
wholebody/policy_contract.json, generated from the checkpoint. tests/test_wholebody.py replays recorded
simulator steps and requires every observation to match.

Nothing here talks to the robot. Interfaces: joint q/dq (policy joint order = G1 29-DoF motor order), pelvis IMU
gyro, pelvis (base) pose in the field/map frame, and a FieldGrid.
"""
import json
from pathlib import Path

import numpy as np

from state_core import Kinematics, JOINT_NAMES

HERE = Path(__file__).resolve().parent
WHOLEBODY = HERE/'wholebody'
FIELD_GROUPS = ((0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 9), (9, 11))   # head, pelvis, torso, feet, hands, knees, shoulders
COMMAND_SITES = [0, 3, 4, 5, 6]                                           # head, feet, hands: the CAT command projection


def load_contract(path=None):
    contract = json.loads(Path(path or WHOLEBODY/'policy_contract.json').read_text())
    if tuple(contract['joint_order']) != JOINT_NAMES:
        raise ValueError("policy joint order differs from the G1 29-DoF order")
    if contract['obs_dim'] != len(contract['feature_names']) or contract['action_dim'] != 29:
        raise ValueError("inconsistent policy contract")
    for key in ('nominal', 'soft_lower', 'soft_upper', 'kp', 'kd', 'torque_limit'):
        contract[key] = np.asarray(contract[key], dtype=float)
        if contract[key].shape != (29,):
            raise ValueError("contract array has the wrong length: "+key)
    contract['upper_action_scales'] = np.asarray(contract['upper_action_scales'], dtype=float)
    contract['hand_radii'] = np.asarray(contract['hand_radii'], dtype=float)
    return contract


def quat_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float)/np.linalg.norm(wxyz)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def navi_rotation(rotation):
    """Heading frame: x = horizontal projection of the body x axis, z up (task_math.navi_rotation)."""
    x = rotation[:, 0].copy(); x[2] = 0.
    x = x/np.linalg.norm(x)
    z = np.array([0., 0., 1.])
    y = np.cross(z, x); y = y/np.linalg.norm(y)
    return np.stack((np.cross(y, z), y, z), -1)


def world_to_navi(navi, values):
    """einsum('ij,...i->...j'): world vectors expressed in the heading frame."""
    return np.asarray(values) @ navi


def unit(values, eps=1e-6):
    return values/(np.linalg.norm(values, axis=-1, keepdims=True)+eps)


def command_from_fields(rtf, cgf, cbf):
    """task_math.compute_cmd_from_rtf: root guidance, projected so no limb moves into its obstacle."""
    v = rtf[:2]*.7
    bhat = cbf[:, :2]/(np.linalg.norm(cbf[:, :2], axis=-1, keepdims=True)+1e-9)
    limits = (bhat*cgf[:, :2]).sum(-1)
    projected = (bhat*v[None]).sum(-1)
    diff = (limits-projected)/((bhat**2).sum(-1)+1e-9)
    delta = np.where((limits > projected)[:, None], diff[:, None]*bhat, 0.)
    velocity = v+delta.mean(0)
    command = np.concatenate(([1.], velocity, [0.]))*.75
    return np.zeros(4) if np.linalg.norm(command[1:]) < .2 else command


def update_phase(command, last_command, stop, phase, phase_dt, gait_bound):
    """task_math.update_phase for a field-driven world (room_enabled False). Returns command, stop, phase, gait."""
    before = stop > 50; during = (not before) and stop > 0; after = (not before) and (not during)
    move2stop = last_command[0] == 1. and command[0] == 0. and before
    if move2stop:
        stop = 50
    if during:
        stop -= 1
    command = (command if before else np.zeros(4)).copy()
    command[0] = 0. if after else 1.
    phase = np.fmod(phase+phase_dt+np.pi, 2*np.pi)-np.pi
    if after:
        phase = np.zeros(2)
    cosine = np.cos(phase)
    gait = np.where(cosine > gait_bound, 1., np.where(cosine < -gait_bound, -1., 0.))
    return command, stop, phase, gait


def motor_targets(action, previous, contract):
    """Legs incremental (0.5), waist/arms absolute around nominal (0.8) rate-limited to 2 rad/s; soft limits."""
    lower, upper, nominal = contract['soft_lower'], contract['soft_upper'], contract['nominal']
    legs = np.clip(previous[:12]+action[:12]*contract['action_scale'], lower[:12], upper[:12])
    requested = nominal[12:]+action[12:]*contract['upper_action_scales']
    delta = contract['upper_target_rate']*contract['control_dt']
    arms = np.clip(np.clip(requested, previous[12:]-delta, previous[12:]+delta), lower[12:], upper[12:])
    return np.concatenate((legs, arms))


def pd_torque(q, dq, targets, contract, feedforward=None):
    torque = contract['kp']*(targets-q)-contract['kd']*dq
    if feedforward is not None:
        torque = torque+feedforward
    return np.clip(torque, -contract['torque_limit'], contract['torque_limit'])


class FieldGrid:
    """Fields on grid POINTS origin + i*dx (training convention): unit guidance gf, unit boundary bf, sdf [m].

    sample() reproduces training exactly, including its interpolation quirk: corners in x-fast order are paired
    with weights in z-fast order (cat_mjlab/fields.py sample_ragged_field keeps it for checkpoint compatibility).
    Points are clamped into the grid (no extrapolation), as in training.
    """
    CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

    def __init__(self, sdf, gf, bf, origin, dx):
        self.sdf = np.asarray(sdf, dtype=np.float32)
        self.gf = np.asarray(gf, dtype=np.float32)
        self.bf = np.asarray(bf, dtype=np.float32)
        self.origin = np.asarray(origin, dtype=float); self.dx = float(dx)
        shape = self.sdf.shape
        if len(shape) != 3 or self.gf.shape != shape+(3,) or self.bf.shape != shape+(3,) or min(shape) < 2:
            raise ValueError("inconsistent field grid")
        self.shape = np.array(shape)

    def sample(self, points):
        points = np.asarray(points, dtype=float)
        xyz = np.minimum(np.maximum((points-self.origin)/self.dx, 0.), self.shape-2)
        base = np.floor(xyz).astype(int); f = xyz-base
        idx = base[:, None, :]+self.CORNERS[None]
        w = [np.stack((1-f[:, i], f[:, i]), -1) for i in range(3)]
        weights = (w[0][:, :, None, None]*w[1][:, None, :, None]*w[2][:, None, None, :]).reshape(len(points), 8)
        gather = lambda a: a[idx[..., 0], idx[..., 1], idx[..., 2]]
        weigh = lambda v: (weights[..., None]*v).sum(1) if v.ndim == 3 else (weights*v).sum(1)
        return weigh(gather(self.gf)), weigh(gather(self.bf)), weigh(gather(self.sdf))


class WholeBodyKinematics(Kinematics):
    """state_core.Kinematics on the TRAINING model (robot.xml): all 13 field sites + body frames for gravity."""

    def __init__(self, contract, path=None):
        super().__init__(path or WHOLEBODY/'robot.xml')
        self.site_names = tuple(contract['field_sites'])+tuple(contract['elbow_sites'])
        self.gravity = np.asarray(contract['gravity'], dtype=float)
        # Masses, inertial offsets and joint anchors/axes for the waist + arm gravity feedforward.
        self.bodies = {}
        def collect(body, parent):
            inertial = body.find('inertial')
            mass = float(inertial.get('mass')) if inertial is not None else 0.
            ipos = np.fromstring(inertial.get('pos', '0 0 0'), sep=' ') if inertial is not None else np.zeros(3)
            joints = [j.get('name') for j in body.findall('joint')]
            self.bodies[body.get('name')] = dict(parent=parent, mass=mass, ipos=ipos, joints=joints)
            for child in body.findall('body'):
                collect(child, body.get('name'))
        collect(self.root, None)
        self.joint_body = {j: name for name, b in self.bodies.items() for j in b['joints']}
        def descendants(name):
            out = [name]
            for child, b in self.bodies.items():
                if b['parent'] == name:
                    out += descendants(child)
            return out
        self.descendants = {name: descendants(name) for name in self.bodies}

    def frames(self, q):
        """Body world frames (pelvis at identity) after joint motion, plus all site transforms."""
        q = np.asarray(q, dtype=float)
        if q.shape != (29,) or not np.isfinite(q).all():
            raise ValueError("invalid model joint vector")
        angles = dict(zip(JOINT_NAMES, q))
        bodies, sites, joints = {}, {}, {}
        def visit(body, parent, is_root=False):
            t = parent.copy() if is_root else parent @ self.transform(body)
            for j in body.findall('joint'):
                a = dict(self.defaults[j.get('class')], **j.attrib)
                axis = np.fromstring(a.get('axis', '0 0 1'), sep=' '); axis = axis/np.linalg.norm(axis)
                anchor = np.fromstring(a.get('pos', '0 0 0'), sep=' ')
                half = .5*angles[j.get('name')]
                r = quat_matrix(np.r_[np.cos(half), axis*np.sin(half)])
                motion = np.eye(4); motion[:3, :3], motion[:3, 3] = r, anchor-r @ anchor
                t = t @ motion
                joints[j.get('name')] = (anchor, axis)
            bodies[body.get('name')] = t
            for site in body.findall('site'):
                sites[site.get('name')] = t @ self.transform(site)
            for child in body.findall('body'):
                visit(child, t)
        visit(self.root, np.eye(4), True)
        return bodies, sites, joints

    def world_sites(self, q, base_pos, base_quat):
        """[13, 3] world positions: 11 field sites then 2 elbow probes."""
        _, sites, _ = self.frames(q)
        base = np.eye(4); base[:3, :3] = quat_matrix(base_quat); base[:3, 3] = base_pos
        return np.array([(base @ sites[name])[:3, 3] for name in self.site_names])

    def upper_gravity(self, q, base_quat):
        """Feedforward torque for joints 12..28 (cat_mjlab/upper_control.py UpperGravity): minus the gravitational
        generalised force of every descendant body's mass about each waist/arm hinge, current orientation."""
        bodies, _, joints = self.frames(q)
        rotation = quat_matrix(base_quat)
        com, mass = {}, {}
        for name, b in self.bodies.items():
            t = bodies[name]
            com[name] = rotation @ (t[:3, :3] @ b['ipos']+t[:3, 3]); mass[name] = b['mass']
        torque = np.zeros(17)
        for k, joint in enumerate(JOINT_NAMES[12:]):
            body = self.joint_body[joint]; t = bodies[body]
            anchor_local, axis_local = joints[joint]
            anchor = rotation @ (t[:3, :3] @ anchor_local+t[:3, 3]); axis = rotation @ (t[:3, :3] @ axis_local)
            members = self.descendants[body]
            total = sum(mass[m] for m in members)
            moment_arm = sum(mass[m]*com[m] for m in members)-total*anchor
            torque[k] = -np.dot(np.cross(moment_arm, self.gravity), axis)
        return torque


def delay_body_pos(base_pos, base_quat, odom_pos, odom_quat, positions):
    """Positions re-expressed from the current base pose onto the HELD odometry pose (task_math.delay_body_pos)."""
    local = (np.asarray(positions)-base_pos) @ quat_matrix(base_quat)
    return odom_pos+local @ quat_matrix(odom_quat).T


class ObservationBuilder:
    """The 226-dim actor observation for field-driven (goal) mode, step by step as in training.

    Call order per control step (50 Hz):  targets = builder.act(action)  ->  hold targets for 20 ms  ->
    obs = builder.observe(q, dq, gyro, base_pos, base_quat, fields).  reset() returns the first observation.
    Training randomises the gait frequency in [1.3, 1.5] Hz per episode; deployment uses `frequency`.
    """

    def __init__(self, contract, kinematics, frequency=None):
        self.c, self.kin = contract, kinematics
        self.frequency = float(np.mean(contract['frequency_range']) if frequency is None else frequency)
        self.dt = contract['control_dt']
        elbow = [n for n in contract['feature_names'] if n.startswith('pf.elbows.')]
        self.elbow_df = np.asarray(contract['elbow_df_index'])
        if len(elbow) != 14:
            raise ValueError("unexpected elbow block")

    # ---------------------------------------------------------------- fields at sites
    def _site_fields(self, fields, positions):
        gf, bf, sdf = fields.sample(positions[:11])
        sdf = sdf.copy(); sdf[5:7] -= self.c['hand_radii']
        return gf, bf, sdf

    def reset(self, q, base_pos, base_quat, fields, dq=None, gyro=None, left_first=True, sites=None, pelvis_rotation=None):
        self.step = 0; self.stop = 100
        self.phase = np.array([0., np.pi]) if left_first else np.array([np.pi, 0.])
        self.phase_dt = 2*np.pi*self.dt*self.frequency
        self.foot_height = self.c['foot_height']
        self.targets = self.c['nominal'].copy(); self.last_act = np.zeros(29)
        self.odom_pos, self.odom_quat = np.asarray(base_pos, float).copy(), np.asarray(base_quat, float).copy()
        self.navi = np.eye(3)                       # reset observation uses the identity heading frame (training)
        self.pelvis_rotation = quat_matrix(base_quat) if pelvis_rotation is None else np.asarray(pelvis_rotation, float)
        positions = self.kin.world_sites(q, base_pos, base_quat) if sites is None else np.asarray(sites, float)
        gf, bf, sdf = self._site_fields(fields, positions)
        gf, bf = unit(gf), unit(bf)
        self.command = command_from_fields(gf[1], gf[COMMAND_SITES], bf[COMMAND_SITES])
        self.command_delay = self.command.copy(); self.last_command = np.zeros(4)
        self.gf_delay, self.bf_delay, self.sdf_delay = gf, bf, sdf
        self.sdf_prev = np.zeros(4)
        return self._observe(q, dq, gyro, base_pos, base_quat, fields, positions)

    def act(self, action):
        """Policy action (tanh output, [-1, 1]) -> joint targets to hold for the next 20 ms."""
        action = np.asarray(action, dtype=float)
        self.targets = motor_targets(action, self.targets, self.c)
        self.last_act = action.copy()
        return self.targets.copy()

    def rearm(self, left_first=True):
        """New goal after a stop: the training stop latch keeps a stopped robot standing for good; restart it with
        anti-phase stepping (as the interactive demo does)."""
        self.stop = 100
        self.phase = np.array([0., np.pi]) if left_first else np.array([np.pi, 0.])

    def observe(self, q, dq, gyro, base_pos, base_quat, fields, sites=None, pelvis_rotation=None):
        """sites / pelvis_rotation: override forward kinematics (replay tests and simulators that report them)."""
        base_pos, base_quat = np.asarray(base_pos, float), np.asarray(base_quat, float)
        self.pelvis_rotation = quat_matrix(base_quat) if pelvis_rotation is None else np.asarray(pelvis_rotation, float)
        self.navi = navi_rotation(self.pelvis_rotation)
        self.last_command = self.command.copy()
        positions = self.kin.world_sites(q, base_pos, base_quat) if sites is None else np.asarray(sites, float)
        gf, bf, sdf = self._site_fields(fields, positions)
        command = command_from_fields(gf[1], gf[COMMAND_SITES], bf[COMMAND_SITES])
        if self.step % self.c['odometry_period_steps'] == 0:
            self.odom_pos, self.odom_quat = base_pos.copy(), base_quat.copy()
        delayed = delay_body_pos(base_pos, base_quat, self.odom_pos, self.odom_quat, positions[:11])
        gfd, bfd, sdfd = self._site_fields(fields, delayed)
        command, self.stop, self.phase, gait = update_phase(command, self.last_command, self.stop, self.phase,
                                                            self.phase_dt, self.c['gait_bound'])
        move = float(command[0] > .5)
        self.gf_delay, self.bf_delay, self.sdf_delay = unit(gfd)*move, unit(bfd), sdfd
        self.command_delay = command_from_fields(self.gf_delay[1], self.gf_delay[COMMAND_SITES], self.bf_delay[COMMAND_SITES])
        self.command = command
        self.step += 1
        return self._observe(q, dq, gyro, base_pos, base_quat, fields, positions)

    def _elbows(self, base_pos, base_quat, fields, positions):
        elbows = delay_body_pos(base_pos, base_quat, self.odom_pos, self.odom_quat, positions[11:13])
        gf, bf, sdf = fields.sample(elbows)
        sdf = sdf-self.c['elbow_sdf_offset']
        move = float(self.step == 0 or self.command[0] > .5)
        gf = gf*move/(np.linalg.norm(gf, axis=-1, keepdims=True)+1e-6)
        bf = unit(bf)
        gf = world_to_navi(self.navi, gf); bf = world_to_navi(self.navi, bf)*(sdf < .5)[:, None]
        return np.concatenate((gf.ravel(), bf.ravel(), np.clip(sdf, -1., .5)))

    def _observe(self, q, dq, gyro, base_pos, base_quat, fields, positions):
        gravity = -self.pelvis_rotation[2, :]
        gyro = np.zeros(3) if gyro is None else np.asarray(gyro, float)
        dq = np.zeros(29) if dq is None else np.asarray(dq, float)
        command = self.command.copy()
        command[1:] = world_to_navi(self.navi, self.command_delay[1:])
        if not self.c['keep_yaw']:
            command[-1] = 0.
        phase = np.concatenate((np.cos(self.phase), np.sin(self.phase)))
        common = np.concatenate((gyro, gravity, np.asarray(q)-self.c['nominal'], dq, self.last_act, self.targets,
                                 command, [self.foot_height], phase))
        gfn = world_to_navi(self.navi, self.gf_delay)
        bfn = world_to_navi(self.navi, self.bf_delay)*(self.sdf_delay < .5)[:, None]
        sdfn = np.clip(self.sdf_delay, -1., .5)
        packed = np.concatenate([np.concatenate((gfn[a:b].ravel(), bfn[a:b].ravel(), sdfn[a:b])) for a, b in FIELD_GROUPS])
        elbows = self._elbows(base_pos, base_quat, fields, positions)
        current = np.concatenate((self.sdf_delay[5:7], elbows[self.elbow_df]))
        rate = np.zeros(4) if self.step == 0 else np.clip((current-self.sdf_prev)/self.dt, -3., 3.)
        self.sdf_prev = current
        obs = np.nan_to_num(np.concatenate((common, packed, elbows, rate))).astype(np.float32)
        if obs.shape != (self.c['obs_dim'],):
            raise ValueError("observation size changed")
        return obs
