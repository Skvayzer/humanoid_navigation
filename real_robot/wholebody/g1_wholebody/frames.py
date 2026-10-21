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
        t_torso_lidar = transform(rpy_matrix(*spec['rpy']), spec['xyz'])
        t_lidar_body = transform(np.eye(3), -np.asarray(lidar_to_imu_t, float))   # IMU origin in the lidar frame
        self.t_torso_body = t_torso_lidar @ t_lidar_body
        self.t_body_torso = inverse(self.t_torso_body)
        self.kin = kinematics
        self.mount = mount

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
