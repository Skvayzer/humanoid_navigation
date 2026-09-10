"""Read-only G1 telemetry validation and upstream-model forward kinematics.

No global pose is guessed: FK output is pelvis-relative unless independently
calibrated. This is a model comparison, not confirmation of robot embodiment.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from perception_core import rotation_matrix

JOINT_NAMES = tuple(
    [side+'_'+joint+'_joint' for side in ('left', 'right')
     for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
    + ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
    + [side+'_'+joint+'_joint' for side in ('left', 'right')
       for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                     'wrist_roll', 'wrist_pitch', 'wrist_yaw')])
SITE_NAMES = ('head', 'imu_in_pelvis', 'imu_in_torso', 'left_foot', 'right_foot',
              'left_palm', 'right_palm', 'left_knee', 'right_knee',
              'left_shoulder', 'right_shoulder')


def telemetry(message):
    if message.mode_pr != 0:
        raise ValueError("unsupported motor coordinates (require PR)")
    if len(message.motor_state) != 35:
        raise ValueError("unexpected LowState motor array")
    q = np.array([s.q for s in message.motor_state[:29]], dtype=float)
    dq = np.array([s.dq for s in message.motor_state[:29]], dtype=float)
    tau = np.array([s.tau_est for s in message.motor_state[:29]], dtype=float)
    quat = np.asarray(message.imu_state.quaternion, dtype=float)
    gyro = np.asarray(message.imu_state.gyroscope, dtype=float)
    acceleration = np.asarray(message.imu_state.accelerometer, dtype=float)
    if quat.shape != (4,) or gyro.shape != (3,) or acceleration.shape != (3,):
        raise ValueError("invalid IMU array lengths")
    if not all(np.isfinite(v).all() for v in (q, dq, tau, quat, gyro, acceleration)):
        raise ValueError("nonfinite telemetry")
    if not .9 < np.linalg.norm(quat) < 1.1 or (np.abs(q) > 6.4).any() or (np.abs(dq) > 100).any():
        raise ValueError("implausible telemetry")
    quat = quat/np.linalg.norm(quat)
    gravity = rotation_matrix(quat[[1, 2, 3, 0]]).T @ np.array([0., 0., -1.])
    return dict(tick=int(message.tick), q=q, dq=dq, tau=tau, quaternion_wxyz=quat,
                gyro=gyro, acceleration=acceleration, gravity=gravity,
                mode_machine=int(message.mode_machine))


class StateBuffer:
    def __init__(self):
        self.state = self.received = self.tick = None

    def receive(self, message, now):
        try:
            value = telemetry(message)
            if not np.isfinite(now):
                raise ValueError("invalid receive time")
            if self.tick is not None:
                delta = (value['tick']-self.tick) & 0xffffffff
                if delta == 0:
                    return False  # duplicates never refresh age
                if delta >= 0x80000000:
                    raise ValueError("LowState tick moved backwards")
            self.state, self.received, self.tick = value, now, value['tick']
            return True
        except ValueError:
            self.state = self.received = self.tick = None
            raise

    def snapshot(self, now, max_age=.25):
        if self.state is None or not 0 <= now-self.received <= max_age:
            raise ValueError("LowState missing or stale")
        return self.state


class Kinematics:
    """Subset of MJCF needed for this pinned, hinge-only G1 model.

    Compared against MuJoCo fixtures. No dynamics engine or controller starts.
    """
    def __init__(self, path=None):
        path = path or Path(__file__).parent/'upstream/g1_mjx_feetonly_torque.xml'
        root = ET.parse(str(path)).getroot()
        if root.find('compiler').get('angle') != 'radian' or root.findall('include'):
            raise ValueError("unsupported MJCF")
        self.defaults = {}
        def defaults(node, inherited):
            value = dict(inherited)
            for j in node.findall('joint'):
                value.update(j.attrib)
            if node.get('class'):
                self.defaults[node.get('class')] = value
            for child in node.findall('default'):
                defaults(child, value)
        defaults(root.find('default'), {})
        self.root = root.find("worldbody/body[@name='pelvis']")
        if tuple(j.get('name') for j in self.root.iter('joint')) != JOINT_NAMES:
            raise ValueError("model joint order changed")
        self.limits = []
        for j in self.root.iter('joint'):
            attrs = dict(self.defaults[j.get('class')], **j.attrib)
            if attrs.get('type', 'hinge') != 'hinge' or float(attrs.get('ref', '0')) != 0:
                raise ValueError("unsupported joint type/reference")
            self.limits.append(np.fromstring(attrs['range'], sep=' '))
        self.limits = np.asarray(self.limits)

    @staticmethod
    def transform(element):
        if any(k in element.attrib for k in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
            raise ValueError("unsupported model orientation")
        t = np.eye(4)
        q = np.fromstring(element.get('quat', '1 0 0 0'), sep=' ')
        t[:3, :3] = rotation_matrix(q[[1, 2, 3, 0]])
        t[:3, 3] = np.fromstring(element.get('pos', '0 0 0'), sep=' ')
        return t

    def sites(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (29,) or not np.isfinite(q).all():
            raise ValueError("invalid model joint vector")
        if (q < self.limits[:, 0]-.1).any() or (q > self.limits[:, 1]+.1).any():
            raise ValueError("joints outside model limits")
        angles = dict(zip(JOINT_NAMES, q))
        sites = {}
        def visit(body, parent, is_root=False):
            t = parent.copy() if is_root else parent @ self.transform(body)
            for j in body.findall('joint'):
                a = dict(self.defaults[j.get('class')], **j.attrib)
                axis = np.fromstring(a.get('axis', '0 0 1'), sep=' ')
                axis /= np.linalg.norm(axis)
                anchor = np.fromstring(a.get('pos', '0 0 0'), sep=' ')
                half_angle = .5*angles[j.get('name')]
                # Use the already-tested quaternion routine on both old Foxy
                # SciPy and current development environments.
                r = rotation_matrix(np.r_[axis*np.sin(half_angle), np.cos(half_angle)])
                motion = np.eye(4)
                motion[:3, :3], motion[:3, 3] = r, anchor-r@anchor
                t = t @ motion
            for site in body.findall('site'):
                sites[site.get('name')] = t @ self.transform(site)
            for child in body.findall('body'):
                visit(child, t)
        visit(self.root, np.eye(4), True)
        return {name: sites[name] for name in SITE_NAMES}
