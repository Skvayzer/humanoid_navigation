"""Pelvis pose in the map from SLAM, the LiDAR mount and the waist joints.

Frames
  body    FAST-LIO's IMU frame: the Mid-360's internal IMU (native axes). SLAM gives T_map_body.
  lidar   the Mid-360 point-cloud frame. FAST-LIO extrinsic: p_body = R_ext p_lidar + T_ext (R_ext = I on the G1).
  torso   torso_link of the policy model (cat/wholebody/robot.xml).
  pelvis  the policy base (root of robot.xml; the pelvis IMU of LowState is aligned with it).

  T_map_pelvis = T_map_body · T_torso_body⁻¹ · T_pelvis_torso(q)⁻¹,   T_torso_body = T_torso_lidar · T_lidar_body

Orientation fusion: roll/pitch of the pelvis come from the pelvis IMU (LowState, 500 Hz, no SLAM latency and no
mount error); heading and position from the chain above. The policy's gravity input is thus independent of the
LiDAR mount calibration; only heading/position (field sampling) depend on it.
"""
import numpy as np

from wholebody_core import quat_matrix

# FAST-LIO extrinsic of the G1 configs (lidar -> IMU): the IMU origin in the lidar frame is -T_ext.
LIDAR_TO_IMU_T = np.array([-0.011, -0.02329, 0.04412])

# Two published Mid-360 mounts on torso_link (they differ by 2.9 deg pitch and 1 cm height; see README).
MOUNTS = {
    'robot_xml': dict(xyz=[0.0002835, 0.00003, 0.4188], rpy=[np.pi, 0.0, 0.0]),            # cat/wholebody/robot.xml
    'unitree_urdf': dict(xyz=[0.0002835, 0.00003, 0.428434], rpy=[np.pi, 0.05112069379091391, 0.0]),  # g1_29dof_rev_1_0.urdf
}


def rpy_matrix(roll, pitch, yaw):
    """URDF convention: fixed axes X, Y, Z -> R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(rotation=None, translation=None):
    t = np.eye(4)
    if rotation is not None:
        t[:3, :3] = rotation
    if translation is not None:
        t[:3, 3] = translation
    return t


def inverse(t):
    out = np.eye(4); r = t[:3, :3]
    out[:3, :3] = r.T; out[:3, 3] = -r.T @ t[:3, 3]
    return out


def pose_matrix(position, quat_wxyz):
    return transform(quat_matrix(quat_wxyz), position)


def matrix_quat(r):
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r))); j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (r[k, j] - r[j, k]) / s; q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s; q[1 + k] = (r[k, i] + r[i, k]) / s
    q = np.asarray(q); q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def heading(r):
    """Yaw of the horizontal projection of the x axis (the policy's navi_rotation convention)."""
    return float(np.arctan2(r[1, 0], r[0, 0]))


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def fuse_orientation(chain_rotation, imu_rotation):
    """Heading from the SLAM chain, tilt (roll/pitch) from the pelvis IMU."""
    tilt = yaw_matrix(-heading(imu_rotation)) @ imu_rotation
    return yaw_matrix(heading(chain_rotation)) @ tilt


class PelvisChain:
    """T_map_body (SLAM) + waist joints -> pelvis pose in the map."""

    def __init__(self, kinematics, mount='unitree_urdf', lidar_to_imu_t=LIDAR_TO_IMU_T):
        spec = MOUNTS[mount] if isinstance(mount, str) else mount
        self.t_torso_lidar0 = transform(rpy_matrix(*spec['rpy']), spec['xyz'])
        self.t_lidar_body = transform(np.eye(3), -np.asarray(lidar_to_imu_t, float))   # IMU origin in the lidar frame
        self.kin = kinematics
        self.mount = mount
        self.set_mount_correction(0., 0.)

    def set_mount_correction(self, roll, pitch):
        """Rotate the LiDAR mount about the torso axes by (roll, pitch) rad: the head (no joint, no encoder) is not
        always where the URDF puts it (2026-11-19: 12 deg pitched forward, then back). Position stays the mount's."""
        self.correction = (float(roll), float(pitch))
        r = rpy_matrix(roll, pitch, 0.)
        t = self.t_torso_lidar0.copy(); t[:3, :3] = r @ t[:3, :3]
        self.t_torso_body = t @ self.t_lidar_body
        self.t_body_torso = inverse(self.t_torso_body)

    def pelvis_torso(self, q):
        bodies, _, _ = self.kin.frames(q)
        return bodies['torso_link']

    def pelvis_from_body(self, t_map_body, q):
        """4x4 T_map_pelvis from the SLAM body pose and the current joint angles (only the waist matters)."""
        return t_map_body @ self.t_body_torso @ inverse(self.pelvis_torso(q))

    def body_from_pelvis(self, t_map_pelvis, q):
        """Inverse map (simulated SLAM in the closed-loop tests)."""
        return t_map_pelvis @ self.pelvis_torso(q) @ self.t_torso_body

    def pelvis_pose(self, t_map_body, q, imu_quat_wxyz=None):
        """(position, quat_wxyz, rotation) of the pelvis in the map; tilt from the pelvis IMU when given."""
        t = self.pelvis_from_body(t_map_body, q)
        rotation = t[:3, :3]
        if imu_quat_wxyz is not None:
            rotation = fuse_orientation(rotation, quat_matrix(imu_quat_wxyz))
        return t[:3, 3].copy(), matrix_quat(rotation), rotation


def tilt_rp(rotation):
    """(roll, pitch) rad of a rotation: the tilt part, heading-free."""
    r = np.asarray(rotation)
    return np.arctan2(r[2, 1], r[2, 2]), np.arcsin(-np.clip(r[2, 0], -1., 1.))


class MountCalibrator:
    """Online LiDAR-mount calibration and head watchdog. While the robot stands still, the torso tilt from SLAM
    (T_map_body through the mount) and from the pelvis IMU + waist FK must agree; the median difference over the last
    second moves the chain's mount correction half-way there (heading is not observable from the IMU: untouched).
    The floor of the SLAM map is level (checked 2026-11-19), so the difference is the head, not SLAM."""

    def __init__(self, chain, gain=.5, min_samples=8, max_deg=20., warn_deg=3.):
        self.chain, self.gain, self.min_samples = chain, gain, min_samples
        self.max, self.warn = np.radians(max_deg), np.radians(warn_deg)
        self.samples = []

    def residual(self, t_map_body, q, imu_quat_wxyz):
        """(roll, pitch) rad: torso tilt from SLAM minus from the IMU, with the current correction."""
        r_slam = (np.asarray(t_map_body) @ self.chain.t_body_torso)[:3, :3]
        r_imu = quat_matrix(imu_quat_wxyz) @ self.chain.pelvis_torso(q)[:3, :3]
        a, b = tilt_rp(r_slam), tilt_rp(r_imu)
        return a[0] - b[0], a[1] - b[1]

    def add(self, t_map_body, q, imu_quat_wxyz):
        self.samples.append(self.residual(t_map_body, q, imu_quat_wxyz))

    def update(self):
        """Apply the last samples (call ~1 Hz). Returns True when the correction changed."""
        if len(self.samples) < self.min_samples:
            self.samples = []
            return False
        dr, dp = np.median(np.array(self.samples), axis=0); self.samples = []
        roll, pitch = self.chain.correction
        roll = float(np.clip(roll + self.gain * dr, -self.max, self.max))
        pitch = float(np.clip(pitch + self.gain * dp, -self.max, self.max))
        self.chain.set_mount_correction(roll, pitch)
        return True

    def warning(self):
        """Text when the head is off the URDF mount by more than warn_deg, else None."""
        roll, pitch = self.chain.correction
        if max(abs(roll), abs(pitch)) <= self.warn:
            return None
        return 'head/LiDAR mount off the URDF: pitch %+.1f, roll %+.1f deg (check the head screws)' % (np.degrees(pitch), np.degrees(roll))
