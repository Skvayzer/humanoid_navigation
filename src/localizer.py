#!/usr/bin/env python3
from __future__ import annotations

"""Map-based localization using scipy ICP on top of Fast-LIO.

Subscribes to /cloud_registered (camera_init frame), aligns to a PLY map,
and publishes the map -> camera_init TF.

Usage (without colcon build):
  python3 localizer.py --ros-args \
    -p map_path:=/path/to/map.ply \
    -p scan_topic:=/cloud_registered \
    -p initial_pose:=[0.0,0.0,0.0,0.0,0.0,0.0]
"""
import time

import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformBroadcaster,
    TransformListener,
)

# --------------------------------------------------------------------------- #
#  Point cloud utilities
# --------------------------------------------------------------------------- #


def pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract XYZ from PointCloud2 as (N, 3) float64 array."""
    n = msg.width * msg.height
    fields = {f.name: f for f in msg.fields}
    step = msg.point_step
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, step)
    pts = np.empty((n, 3), dtype=np.float32)
    for i, ax in enumerate(("x", "y", "z")):
        off = fields[ax].offset
        pts[:, i] = np.frombuffer(
            np.ascontiguousarray(raw[:, off : off + 4]).tobytes(), dtype=np.float32
        )
    valid = pts[np.isfinite(pts).all(axis=1)]
    return valid.astype(np.float64)


def xyz_to_pc2(pts: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Convert Nx3 array to PointCloud2 message."""
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(pts)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(pts)
    msg.is_dense = True
    msg.data = pts.astype(np.float32).tobytes()
    return msg


# --------------------------------------------------------------------------- #
#  Scipy ICP (Open3D registration crashes on ARM64 / Open3D 0.18)
# --------------------------------------------------------------------------- #


def _transform_pts(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ h.T).T[:, :3]


def _query_tree(tree: cKDTree, points: np.ndarray):
    """Query with two workers on both old (G1/Focal) and new SciPy APIs."""
    try:
        return tree.query(points, k=1, workers=2)
    except TypeError:
        return tree.query(points, k=1, n_jobs=2)


def icp(
    src_pts: np.ndarray,
    tgt_tree: cKDTree,
    tgt_pts: np.ndarray,
    init_T: np.ndarray,
    max_dist: float,
    iterations: int = 30,
) -> tuple[np.ndarray, float, float]:
    """Point-to-point ICP via scipy KDTree + SVD.

    Returns (T_4x4, fitness, rmse).
    fitness = fraction of src points within max_dist of tgt after alignment.
    """
    T = init_T.copy()
    for _ in range(iterations):
        src_t = _transform_pts(src_pts, T)
        dists, idxs = _query_tree(tgt_tree, src_t)
        mask = dists < max_dist
        if mask.sum() < 20:
            break
        src_m = src_t[mask]
        tgt_m = tgt_pts[idxs[mask]]

        sc, tc = src_m.mean(0), tgt_m.mean(0)
        H = (src_m - sc).T @ (tgt_m - tc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = tc - R @ sc

        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t
        T = dT @ T

    src_t = _transform_pts(src_pts, T)
    dists, _ = _query_tree(tgt_tree, src_t)
    mask = dists < max_dist
    fitness = float(mask.sum()) / max(len(src_pts), 1)
    rmse = float(np.sqrt((dists[mask] ** 2).mean())) if mask.sum() > 0 else float("inf")
    return T, fitness, rmse


def global_icp(
    src_pts: np.ndarray,
    tgt_tree: cKDTree,
    tgt_pts: np.ndarray,
    hint_T: np.ndarray,
    max_dist: float,
) -> tuple[np.ndarray, float, float]:
    """Try ICP from hint + 8 yaw rotations; return best result."""
    best_T, best_fit, best_rmse = hint_T.copy(), -1.0, float("inf")

    src_centre = _transform_pts(src_pts, hint_T).mean(0)

    for yaw_deg in range(0, 360, 5):
        yaw_R = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
        dT = np.eye(4)
        dT[:3, :3] = yaw_R
        dT[:3, 3] = src_centre - yaw_R @ src_centre
        T_init = dT @ hint_T

        T, fit, rmse = icp(src_pts, tgt_tree, tgt_pts, T_init, max_dist)
        if fit > best_fit:
            best_T, best_fit, best_rmse = T, fit, rmse

    return best_T, best_fit, best_rmse


def icp_2d(
    src_xy: np.ndarray,
    tgt_tree_2d: cKDTree,
    tgt_pts_2d: np.ndarray,
    init_T: np.ndarray,
    max_dist: float,
    iterations: int = 30,
) -> tuple[np.ndarray, float, float]:
    """2D point-to-point ICP (XY + yaw only) via scipy KDTree + SVD.

    Solves only x, y, yaw — prevents false-positive Z-drift in 3D ICP.
    Returns a 4x4 transform (Z translation kept from init_T, roll/pitch = 0).
    """
    T2 = np.eye(3)
    T2[:2, :2] = init_T[:2, :2]
    T2[:2, 2] = init_T[:2, 3]

    for _ in range(iterations):
        sh = np.hstack([src_xy, np.ones((len(src_xy), 1))])
        st = (T2 @ sh.T).T[:, :2]
        dists, idxs = _query_tree(tgt_tree_2d, st)
        mask = dists < max_dist
        if mask.sum() < 20:
            break
        sm, tm = st[mask], tgt_pts_2d[idxs[mask]]
        sc, tc = sm.mean(0), tm.mean(0)
        H = (sm - sc).T @ (tm - tc)
        U, _, Vt = np.linalg.svd(H)
        R2 = Vt.T @ U.T
        if np.linalg.det(R2) < 0:
            Vt[-1] *= -1
            R2 = Vt.T @ U.T
        dT2 = np.eye(3)
        dT2[:2, :2] = R2
        dT2[:2, 2] = tm.mean(0) - R2 @ sm.mean(0)
        T2 = dT2 @ T2

    sh = np.hstack([src_xy, np.ones((len(src_xy), 1))])
    st = (T2 @ sh.T).T[:, :2]
    dists, _ = _query_tree(tgt_tree_2d, st)
    mask = dists < max_dist
    fitness = float(mask.sum()) / max(len(src_xy), 1)
    rmse = float(np.sqrt((dists[mask] ** 2).mean())) if mask.sum() > 0 else float("inf")

    T4 = np.eye(4)
    T4[:2, :2] = T2[:2, :2]
    T4[:2, 3] = T2[:2, 2]
    T4[2, 3] = init_T[2, 3]  # preserve Z from hint (don't drift vertically)
    return T4, fitness, rmse


def global_icp_2d(
    src_pts: np.ndarray,
    tgt_tree_2d: cKDTree,
    tgt_pts_2d: np.ndarray,
    hint_T: np.ndarray,
    max_dist: float,
) -> tuple[np.ndarray, float, float]:
    """Try 2D ICP from hint + 8 yaw rotations; return best result."""
    src_xy = src_pts[:, :2]
    T2_hint = np.eye(3)
    T2_hint[:2, :2] = hint_T[:2, :2]
    T2_hint[:2, 2] = hint_T[:2, 3]
    sh = np.hstack([src_xy, np.ones((len(src_xy), 1))])
    src_centre_2d = (T2_hint @ sh.T).T[:, :2].mean(0)

    best_T, best_fit, best_rmse = hint_T.copy(), -1.0, float("inf")
    for yaw_deg in range(0, 360, 5):
        R2 = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()[:2, :2]
        T_init = hint_T.copy()
        T_init[:2, :2] = R2 @ hint_T[:2, :2]
        T_init[:2, 3] = src_centre_2d - R2 @ src_centre_2d + R2 @ hint_T[:2, 3]
        T, fit, rmse = icp_2d(src_xy, tgt_tree_2d, tgt_pts_2d, T_init, max_dist)
        if fit > best_fit:
            best_T, best_fit, best_rmse = T, fit, rmse

    return best_T, best_fit, best_rmse


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  Open3D-based ICP (point-to-plane) + FPFH+RANSAC global registration
#  Same primitives the inauguration demo's `open3d_loc` uses, but called from
#  Python so we don't depend on the C++ Open3D dev package.
# --------------------------------------------------------------------------- #


def _project_yaw_only(T: np.ndarray) -> np.ndarray:
    """Return a copy of T whose rotation is the pure-yaw rotation closest to
    T's full rotation. Roll/pitch are zeroed. Used to force ICP into the
    physically-valid subspace (both map and scan are already gravity-aligned)."""
    yaw = float(Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)[2])
    R_yaw = Rotation.from_euler("z", yaw).as_matrix()
    out = T.copy()
    out[:3, :3] = R_yaw
    return out


def icp_point_to_plane(
    src_pts: np.ndarray,
    tgt_tree: cKDTree,
    tgt_pts: np.ndarray,
    tgt_normals: np.ndarray,
    init_T: np.ndarray,
    max_dist: float,
    iterations: int = 30,
) -> tuple[np.ndarray, float, float]:
    """Hand-rolled point-to-plane ICP. Same algorithm Open3D uses internally,
    but in pure numpy because Open3D's `registration_icp` segfaults on this
    ARM64 build. Requires `tgt_normals` precomputed for each target point.

    Per iteration, minimises sum_i (n_i · (T·p_i - q_i))^2 by small-angle
    linearisation. We solve only for yaw + (tx, ty, tz) (4-DOF) — roll/pitch
    are locked at 0 because both scan and map are gravity-aligned. The input
    init_T is also projected to yaw-only so any flipped pose from FPFH+RANSAC
    upstream is corrected before refinement.
    """
    T = _project_yaw_only(init_T)
    for _ in range(iterations):
        src_t = _transform_pts(src_pts, T)
        dists, idxs = _query_tree(tgt_tree, src_t)
        mask = dists < max_dist
        if mask.sum() < 20:
            break
        p = src_t[mask]
        q = tgt_pts[idxs[mask]]
        n = tgt_normals[idxs[mask]]
        # 3-DOF point-to-plane: yaw + (tx, ty). Roll/pitch are locked at 0
        # because both scan and map are gravity-aligned; Z is locked because
        # both have their floors at known heights (map Z=0, scan Z=LiDAR
        # mount). The free DOF for a wheeled / footed indoor robot are XY
        # translation and yaw rotation only.
        pcn = np.cross(p, n)  # N x 3
        A = np.column_stack([pcn[:, 2], n[:, 0], n[:, 1]])  # N x 3: yaw, tx, ty
        b = np.einsum("ij,ij->i", q - p, n)  # N
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        alpha = np.array([0.0, 0.0, float(x[0])])  # roll=pitch=0, yaw=x[0]
        t = np.array([float(x[1]), float(x[2]), 0.0])  # tx, ty, tz=0
        R_inc = Rotation.from_rotvec(alpha).as_matrix()
        dT = np.eye(4)
        dT[:3, :3] = R_inc
        dT[:3, 3] = t
        T = dT @ T

    src_t = _transform_pts(src_pts, T)
    dists, _ = _query_tree(tgt_tree, src_t)
    mask = dists < max_dist
    fitness = float(mask.sum()) / max(len(src_pts), 1)
    rmse = float(np.sqrt((dists[mask] ** 2).mean())) if mask.sum() > 0 else float("inf")
    return T, fitness, rmse


def multi_scale_icp_point_to_plane(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    tgt_normals: np.ndarray,
    init_T: np.ndarray,
    voxel_sizes: tuple,
    max_dist_per_voxel_mult: float = 2.5,
    iterations_per_scale: int = 15,
) -> tuple[np.ndarray, float, float]:
    """Cascade ICP coarse→fine. Voxels iterate from largest to smallest.
    Each scale uses max_dist = voxel * mult so coarse step is robust to
    far-from-truth init and fine step tightens to sub-voxel residual.

    Same approach as Open3D's RegistrationMultiScaleICP (used by inauguration
    demo's open3d_loc). Loops 3 ICP runs at decreasing voxel sizes."""
    T = init_T.copy()
    fit, rmse = -1.0, float("inf")
    for v in voxel_sizes:
        if v <= 0:
            continue
        src_ds = voxel_downsample(src_pts, v)
        # Downsample target (with normals) via Open3D, preserves per-cell normals
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(tgt_pts, dtype=np.float64)
        )
        tgt_pcd.normals = o3d.utility.Vector3dVector(
            np.ascontiguousarray(tgt_normals, dtype=np.float64)
        )
        tgt_down = tgt_pcd.voxel_down_sample(v)
        tp = np.asarray(tgt_down.points)
        tn = np.asarray(tgt_down.normals)
        if len(tp) < 20 or len(src_ds) < 20:
            continue
        tt = cKDTree(tp)
        T, fit, rmse = icp_point_to_plane(
            src_ds,
            tt,
            tp,
            tn,
            T,
            max_dist=v * max_dist_per_voxel_mult,
            iterations=iterations_per_scale,
        )
    return T, fit, rmse


def global_icp_point_to_plane(
    src_pts: np.ndarray,
    tgt_tree: cKDTree,
    tgt_pts: np.ndarray,
    tgt_normals: np.ndarray,
    hint_T: np.ndarray,
    max_dist: float,
    max_drift_from_hint: float = 3.0,
) -> tuple[np.ndarray, float, float]:
    """36-yaw global ICP using point-to-plane. Returns the best of all seeds.

    Results whose XY translation drifts more than `max_drift_from_hint` metres
    from the hint are discarded — keeps the search bounded near the user's
    initial_pose hint instead of letting ICP fly off across the map.
    """
    best_T, best_fit, best_rmse = hint_T.copy(), -1.0, float("inf")
    src_centre = _transform_pts(src_pts, hint_T).mean(0)
    hint_xy = hint_T[:2, 3]
    for yaw_deg in range(0, 360, 5):
        yaw_R = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
        dT = np.eye(4)
        dT[:3, :3] = yaw_R
        dT[:3, 3] = src_centre - yaw_R @ src_centre
        T_init = dT @ hint_T
        T, fit, rmse = icp_point_to_plane(
            src_pts, tgt_tree, tgt_pts, tgt_normals, T_init, max_dist
        )
        if float(np.linalg.norm(T[:2, 3] - hint_xy)) > max_drift_from_hint:
            continue
        if fit > best_fit:
            best_T, best_fit, best_rmse = T, fit, rmse
    return best_T, best_fit, best_rmse


class KalmanScalar:
    """1-D Kalman filter with constant-position model. Used by the inauguration
    pipeline to smooth each pose component (x, y, z, yaw) independently."""

    def __init__(self, process_var: float, meas_var: float, x0: float = 0.0):
        self.x = float(x0)
        self.P = 1.0
        self.Q = float(process_var)
        self.R = float(meas_var)

    def reset(self, x0: float) -> None:
        self.x = float(x0)
        self.P = 1.0

    def step(self, z: float) -> float:
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (float(z) - self.x)
        self.P = (1.0 - K) * self.P
        return self.x


def fpfh_ransac_global(
    src_pts: np.ndarray,
    src_normals: np.ndarray,
    tgt_pcd: o3d.geometry.PointCloud,
    tgt_fpfh,
    feature_radius: float,
    distance_threshold: float,
) -> tuple[np.ndarray, float, float]:
    """FPFH+RANSAC hint-free global registration. Returns (T, fitness, rmse).

    Used for cold-start and lost-recovery. Open3D's
    `registration_ransac_based_on_feature_matching` works on this ARM64
    build (unlike `registration_icp`), so we call it directly.
    """
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(src_pts, dtype=np.float64)
    )
    src_pcd.normals = o3d.utility.Vector3dVector(
        np.ascontiguousarray(src_normals, dtype=np.float64)
    )
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd,
        tgt_pcd,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return (
        np.asarray(result.transformation),
        float(result.fitness),
        float(result.inlier_rmse),
    )


def estimate_normals_safe(pts: np.ndarray, radius: float) -> np.ndarray:
    """Estimate per-point normals via Open3D, then flip toward +Z manually
    (Open3D 0.18 on this ARM64 build segfaults inside
    `orient_normals_to_align_with_direction`, so we do it ourselves)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    normals = np.asarray(pcd.normals).copy()
    flip = normals[:, 2] < 0.0
    normals[flip] *= -1.0
    return normals


def hint_to_matrix(pose: list) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def matrix_to_ros_tf(T: np.ndarray, stamp, parent: str, child: str) -> TransformStamped:
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = parent
    ts.child_frame_id = child
    ts.transform.translation.x = float(T[0, 3])
    ts.transform.translation.y = float(T[1, 3])
    ts.transform.translation.z = float(T[2, 3])
    ts.transform.rotation.x = float(quat[0])
    ts.transform.rotation.y = float(quat[1])
    ts.transform.rotation.z = float(quat[2])
    ts.transform.rotation.w = float(quat[3])
    return ts


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Voxel grid downsampling via Open3D (stable on ARM64)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64))
    down = pcd.voxel_down_sample(voxel)
    return np.asarray(down.points, dtype=np.float64)


def fit_floor_normal(
    pts: np.ndarray,
    z_quantile: float = 0.25,
    dist_thresh: float = 0.05,
) -> tuple[np.ndarray, float]:
    """RANSAC plane fit on the lowest z_quantile fraction of points.

    Returns (unit_normal, plane_offset_d). Normal is forced to have positive Z
    so the floor's "up" matches +Z. plane: n . p + d = 0.
    """
    if len(pts) < 200:
        return np.array([0.0, 0.0, 1.0]), 0.0
    z_thresh = np.quantile(pts[:, 2], z_quantile)
    cands = pts[pts[:, 2] <= z_thresh]
    if len(cands) < 200:
        return np.array([0.0, 0.0, 1.0]), 0.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(cands, dtype=np.float64)
    )
    plane, _ = pcd.segment_plane(
        distance_threshold=dist_thresh, ransac_n=3, num_iterations=500
    )
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    if n[2] < 0.0:
        n = -n
        d = -d
    nrm = float(np.linalg.norm(n))
    return n / nrm, float(d) / nrm


def rotation_to_align(n_from: np.ndarray, n_to: np.ndarray) -> np.ndarray:
    """3x3 rotation taking unit vector n_from onto n_to (Rodrigues)."""
    v = np.cross(n_from, n_to)
    s = float(np.linalg.norm(v))
    c = float(np.dot(n_from, n_to))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64
    )
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def gravity_align_floor(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate map so the dominant floor plane normal becomes +Z, then shift floor to Z=0.

    Returns (aligned_pts, R_floor, floor_z_shift). R_floor takes raw_pts to gravity-aligned;
    floor_z_shift is the Z offset removed after rotation.
    """
    n_floor, _ = fit_floor_normal(pts)
    R_floor = rotation_to_align(n_floor, np.array([0.0, 0.0, 1.0]))
    aligned = np.ascontiguousarray((R_floor @ pts.T).T)
    z_floor = float(np.quantile(aligned[:, 2], 0.05))
    aligned[:, 2] -= z_floor
    return aligned, R_floor, z_floor


# --------------------------------------------------------------------------- #
#  ROS2 Node
# --------------------------------------------------------------------------- #


class LocalizerNode(Node):
    """Localizes Fast-LIO odometry against a pre-built PLY map."""

    def __init__(self):
        super().__init__("map_localizer")

        self.declare_parameter("map_path", "")
        self.declare_parameter("scan_topic", "/cloud_registered")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "world")
        self.declare_parameter("voxel_map", 0.2)  # ICP target resolution
        self.declare_parameter("voxel_scan", 0.3)  # scan downsampling
        self.declare_parameter("voxel_display", 0.1)  # map cloud for Foxglove
        self.declare_parameter("icp_max_dist", 0.5)
        self.declare_parameter("min_fitness", 0.7)
        self.declare_parameter(
            "min_fitness_reloc", 0.6
        )  # stricter floor for accepting a global re-loc / first-localize; below this is rejected
        self.declare_parameter(
            "max_dpose_per_tick", 1.0
        )  # m; reject jumps faster than this/loc_period
        self.declare_parameter(
            "max_rmse", 0.15
        )  # m; reject high-RMSE matches even if fitness passes
        self.declare_parameter(
            "relocalize_after_n_losses", 3
        )  # global re-ICP after N consecutive bad ticks
        self.declare_parameter(
            "max_ceiling_above_floor", 2.5
        )  # m; drop scan points more than this above the floor (the prebuilt map has no ceiling — low-mounted mapping LiDAR — so high scan points kill fitness)
        self.declare_parameter(
            "use_3d_icp", True
        )  # True=6-DOF (icp), False=XY+yaw only (icp_2d)
        self.declare_parameter(
            "use_kalman", True
        )  # True = inauguration pipeline's Kalman smoothing
        self.declare_parameter(
            "kalman_process_var", 0.001
        )  # matches open3d_loc kalman_processVar2
        self.declare_parameter(
            "kalman_meas_var", 0.02
        )  # matches open3d_loc kalman_estimatedMeasVar2
        self.declare_parameter(
            "use_fpfh_ransac", True
        )  # True = use FPFH+RANSAC for cold-start/recovery
        self.declare_parameter(
            "dis_updatemap", 3.5
        )  # rebuild local map subset after pose drift this far
        self.declare_parameter(
            "local_map_radius", 6.0
        )  # half-edge of local map subset cube (must be > dis_updatemap)
        self.declare_parameter(
            "confidence_th", 0.7
        )  # fitness threshold to update Kalman (matches open3d_loc confidence_loc_th)
        self.declare_parameter(
            "max_roll_pitch_deg", 10.0
        )  # reject ICP results whose rotation has more roll/pitch than this
        self.declare_parameter(
            "max_init_dist_from_hint", 3.0
        )  # cold-start reject if found pose is farther than this from initial_pose
        self.declare_parameter(
            "max_recovery_drift_from_prior", 1.0
        )  # recovery: reject FPFH/global ICP result if farther than this from last good pose
        self.declare_parameter(
            "use_multi_scale_icp", True
        )  # cascade ICP coarse→fine voxels (matches inauguration RegistrationMultiScaleICP)
        self.declare_parameter(
            "multi_scale_voxels", [0.6, 0.4, 0.2]
        )  # voxel sizes, coarse to fine — cold-start / global recovery only
        self.declare_parameter(
            "track_multi_scale_voxels", [0.3, 0.15]
        )  # fine-only cascade for per-tick tracking (no coarse stage → can't jump to a far basin)
        self.declare_parameter(
            "odom_topic", "/Odometry"
        )  # Fast-LIO odometry — used as the per-tick trust/speed model
        self.declare_parameter(
            "max_map_drift_rate", 0.15
        )  # m/s; T_map_cam relates two fixed frames, so it may only move at Fast-LIO's drift rate
        self.declare_parameter(
            "drift_rate_speed_gain", 0.30
        )  # extra drift budget (m per m/s of robot speed) — looser bound while walking
        self.declare_parameter(
            "jump_fitness_margin", 0.08
        )  # a jump past the drift budget is only accepted if fitness beats the prior pose by this much
        self.declare_parameter("loc_period", 1.0)
        self.declare_parameter("freeze_after_init", False)
        self.declare_parameter("map_cloud_topic", "/map_cloud")
        self.declare_parameter("initial_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        map_path = self.get_parameter("map_path").value
        scan_topic = self.get_parameter("scan_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        voxel_map = float(self.get_parameter("voxel_map").value)
        voxel_scan = float(self.get_parameter("voxel_scan").value)
        voxel_display = float(self.get_parameter("voxel_display").value)
        self.icp_max_dist = float(self.get_parameter("icp_max_dist").value)
        self.min_fitness = float(self.get_parameter("min_fitness").value)
        self.min_fitness_reloc = float(self.get_parameter("min_fitness_reloc").value)
        self.max_dpose_per_tick = float(self.get_parameter("max_dpose_per_tick").value)
        self.max_rmse = float(self.get_parameter("max_rmse").value)
        self.relocalize_after_n_losses = int(
            self.get_parameter("relocalize_after_n_losses").value
        )
        self._max_ceiling_above_floor = float(
            self.get_parameter("max_ceiling_above_floor").value
        )
        self._use_3d_icp = bool(self.get_parameter("use_3d_icp").value)
        self._use_kalman = bool(self.get_parameter("use_kalman").value)
        kf_proc = float(self.get_parameter("kalman_process_var").value)
        kf_meas = float(self.get_parameter("kalman_meas_var").value)
        self._use_fpfh_ransac = bool(self.get_parameter("use_fpfh_ransac").value)
        self._dis_updatemap = float(self.get_parameter("dis_updatemap").value)
        self._local_map_radius = float(self.get_parameter("local_map_radius").value)
        self._confidence_th = float(self.get_parameter("confidence_th").value)
        self._max_roll_pitch_deg = float(self.get_parameter("max_roll_pitch_deg").value)
        self._max_init_dist_from_hint = float(
            self.get_parameter("max_init_dist_from_hint").value
        )
        self._max_recovery_drift_from_prior = float(
            self.get_parameter("max_recovery_drift_from_prior").value
        )
        self._use_multi_scale_icp = bool(
            self.get_parameter("use_multi_scale_icp").value
        )
        self._multi_scale_voxels = tuple(
            float(v) for v in self.get_parameter("multi_scale_voxels").value
        )
        self._track_voxels = tuple(
            float(v) for v in self.get_parameter("track_multi_scale_voxels").value
        )
        odom_topic = self.get_parameter("odom_topic").value
        self._max_map_drift_rate = float(self.get_parameter("max_map_drift_rate").value)
        self._drift_rate_speed_gain = float(
            self.get_parameter("drift_rate_speed_gain").value
        )
        self._jump_fitness_margin = float(
            self.get_parameter("jump_fitness_margin").value
        )
        loc_period = float(self.get_parameter("loc_period").value)
        self._loc_period = loc_period
        self._freeze_after_init = bool(self.get_parameter("freeze_after_init").value)
        map_cloud_topic = self.get_parameter("map_cloud_topic").value
        init_pose = list(self.get_parameter("initial_pose").value)
        self._init_pose_xy = np.array([init_pose[0], init_pose[1]], dtype=float)

        if not map_path:
            self.get_logger().fatal("map_path parameter is required")
            raise SystemExit(1)

        # Load map
        self.get_logger().info(f"Loading map: {map_path}")
        raw = o3d.io.read_point_cloud(map_path)
        self.get_logger().info(f"Map loaded: {len(raw.points)} raw points")

        raw_pts = np.asarray(raw.points)
        # Gravity-align the PLY map via its own floor plane: makes floor horizontal
        # in the map frame regardless of the original mapping session's CI tilt.
        aligned_pts, R_floor, z_floor_shift = gravity_align_floor(raw_pts)
        self._R_map_floor = R_floor
        self._tgt_pts_raw = aligned_pts
        self._voxel_map_size = voxel_map
        self._voxel_display_size = voxel_display
        # Clip extreme Z outliers (ceiling artefacts, sub-floor noise)
        z_mask = (aligned_pts[:, 2] >= -0.3) & (aligned_pts[:, 2] <= 5.0)
        icp_pts = aligned_pts[z_mask]
        self._tgt_pts = voxel_downsample(icp_pts, voxel_map)
        self._tgt_tree = cKDTree(self._tgt_pts)
        tgt_xy = self._tgt_pts[:, :2]
        self._tgt_tree_2d = cKDTree(tgt_xy)
        self._tgt_pts_2d = tgt_xy
        self._display_pts = voxel_downsample(aligned_pts, voxel_display)
        self._voxel_scan = voxel_scan
        # Per-point normals on the downsampled map (used when use_3d_icp=True
        # for point-to-plane ICP). Open3D estimates them; we skip its
        # orient-by-direction step which segfaults on ARM64 0.18.
        self._tgt_normals = estimate_normals_safe(self._tgt_pts, radius=voxel_map * 2.5)

        # ---------- Inauguration pipeline additions ----------
        # Full-map (un-clipped) version for cold-start FPFH+RANSAC and local
        # subset rebuilding. Always keep the un-clipped aligned map around.
        self._tgt_pts_full = aligned_pts
        self._tgt_normals_full = estimate_normals_safe(
            aligned_pts, radius=voxel_map * 2.5
        )
        # FPFH on the downsampled-and-clipped target — precomputed once for
        # hint-free recovery via RANSAC.
        if self._use_fpfh_ransac:
            self.get_logger().info("Precomputing FPFH features on map (one-time)…")
            self._tgt_pcd_o3d = o3d.geometry.PointCloud()
            self._tgt_pcd_o3d.points = o3d.utility.Vector3dVector(self._tgt_pts)
            self._tgt_pcd_o3d.normals = o3d.utility.Vector3dVector(self._tgt_normals)
            self._tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                self._tgt_pcd_o3d,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_map * 5.0, max_nn=100
                ),
            )
            self.get_logger().info(
                f"FPFH features ready ({np.asarray(self._tgt_fpfh.data).shape})"
            )
        else:
            self._tgt_pcd_o3d = None
            self._tgt_fpfh = None

        # Kalman filters for pose smoothing — one per pose component
        self._kf_x = KalmanScalar(kf_proc, kf_meas, init_pose[0])
        self._kf_y = KalmanScalar(kf_proc, kf_meas, init_pose[1])
        self._kf_z = KalmanScalar(kf_proc, kf_meas, init_pose[2])
        self._kf_yaw = KalmanScalar(kf_proc, kf_meas, float(np.deg2rad(init_pose[5])))

        # Local map subset state: rebuilt every time the pose moves dis_updatemap
        self._local_map_centre: np.ndarray | None = None  # xy of last subset rebuild
        _floor_normal_world = R_floor @ np.array([0.0, 0.0, 1.0])  # unused, sanity
        self.get_logger().info(
            f"Map gravity-aligned via floor plane — "
            f"R_floor det={np.linalg.det(R_floor):.3f} "
            f"z_shift={z_floor_shift:.3f}m, "
            f"ICP pts={len(self._tgt_pts)} display pts={len(self._display_pts)} "
            f"Z:[{self._tgt_pts[:, 2].min():.2f},{self._tgt_pts[:, 2].max():.2f}]"
        )

        self.T_map_cam = hint_to_matrix(init_pose)
        self._T_published: np.ndarray = hint_to_matrix(init_pose)
        self._tf_smooth_alpha = 0.3  # EMA weight for new ICP result (lower = smoother)
        self._initialized = False
        self._latest_cloud: np.ndarray | None = None
        self._latest_stamp = (
            None  # stamp from latest cloud msg — keeps TF in Fast-LIO time domain
        )
        self._cam_to_world_R: np.ndarray | None = None  # set once gravity TF arrives
        self._floor_z_world: float = (
            0.0  # floor height in world frame (for Z=0 alignment)
        )
        self._lost_count: int = (
            0  # consecutive bad-track ticks (triggers re-localization)
        )
        self._robot_speed: float = (
            0.0  # m/s — derived from /Odometry pose deltas (Fast-LIO leaves twist empty)
        )
        self._last_odom_pos: np.ndarray | None = None
        self._last_odom_t: float | None = None
        # Ticks after a (re)lock during which the loose ceiling is used instead
        # of the tight drift budget — lets ICP settle before the gate clamps.
        self._settle_ticks: int = 0
        self._settle_ticks_after_lock: int = 10

        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)
        self._tf_br = TransformBroadcaster(self)
        self._static_tf_br = StaticTransformBroadcaster(self)
        self._frozen_static_published = False

        # TRANSIENT_LOCAL = latched: late subscribers get the last message instantly
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._map_pub = self.create_publisher(PointCloud2, map_cloud_topic, latched_qos)
        self._filtered_scan_pub = self.create_publisher(
            PointCloud2, "/filtered_scan", 10
        )

        self._sub = self.create_subscription(PointCloud2, scan_topic, self._cloud_cb, 5)
        self._odom_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_cb, 5
        )
        self._timer = self.create_timer(loc_period, self._localize)
        self._map_published = False

        self.get_logger().info(
            f"Localizer ready — {scan_topic} -> {self.map_frame}/{self.odom_frame}, "
            f"map cloud on {map_cloud_topic} (latched)"
        )

    # ----------------------------------------------------------------------- #

    def _cloud_cb(self, msg: PointCloud2):
        self._latest_cloud = pc2_to_xyz(msg)
        self._latest_stamp = msg.header.stamp

    def _odom_cb(self, msg: Odometry):
        """Cache robot body speed from Fast-LIO /Odometry. Fast-LIO leaves the
        twist field zero, so speed is differentiated from consecutive poses.
        Not a pose prior — the robot's motion is already in /cloud_registered —
        but the trust model: how fast `T_map_cam` is allowed to drift."""
        p = msg.pose.pose.position
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_odom_pos is not None and self._last_odom_t is not None:
            dt = t - self._last_odom_t
            if dt > 1e-3:
                self._robot_speed = float(
                    np.linalg.norm(pos - self._last_odom_pos) / dt
                )
        self._last_odom_pos = pos
        self._last_odom_t = t

    def _eval_fitness(self, scan: np.ndarray, T: np.ndarray) -> float:
        """Inlier fraction of `scan` against the current target at pose `T`,
        without running ICP. Used to decide if a far ICP jump is justified."""
        src_t = _transform_pts(scan, T)
        dists, _ = _query_tree(self._tgt_tree, src_t)
        return float((dists < self.icp_max_dist).sum()) / max(len(scan), 1)

    def _try_get_gravity_tf(self):
        """Cache the world→camera_init rotation; only used to bring scans Z-up.

        Map is already gravity-aligned via floor-plane fit at load time, so this
        rotation is *not* applied to the map.
        """
        try:
            tf = self._tf_buf.lookup_transform(
                "world", "camera_init", rclpy.time.Time()
            )
            q = tf.transform.rotation
            self._cam_to_world_R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            self.get_logger().info("Gravity TF acquired — scan rotation enabled.")
        except Exception:
            pass

    def _localize(self):
        # Fast path: when frozen, skip all scan processing (saves CPU)
        if self._freeze_after_init and self._initialized:
            # Frozen pose is constant — publish map->world ONCE as a static
            # (latched, non-expiring) transform. A dynamic stamp can freeze if
            # the cloud feed stalls, expire from the TF buffer, and make the
            # whole map frame "not found" downstream. Static never expires.
            if not self._frozen_static_published:
                ts = matrix_to_ros_tf(
                    self._T_published,
                    self.get_clock().now().to_msg(),
                    self.map_frame,
                    self.odom_frame,
                )
                self._static_tf_br.sendTransform(ts)
                self._frozen_static_published = True
            return
        # Publish latched map cloud once on first tick (map is already aligned in __init__)
        if not self._map_published:
            self._publish_map_cloud()
            self._map_published = True

        if self._cam_to_world_R is None:
            self._try_get_gravity_tf()
            if self._cam_to_world_R is None:
                return  # need gravity TF to bring scan Z-up before ICP

        if self._latest_cloud is None or len(self._latest_cloud) < 200:
            return

        # Rotate scan from camera_init -> world (Z-up) before ICP
        pts = np.ascontiguousarray((self._cam_to_world_R @ self._latest_cloud.T).T)
        # Ceiling filter — the prebuilt map has no ceiling (low-mounted mapping
        # LiDAR), so scan points above the map's Z range have no map
        # correspondence and only drag fitness down. Robust floor estimate from
        # the 5% Z quantile; drop anything more than `max_ceiling_above_floor`
        # metres above it.
        if len(pts):
            z_floor_est = float(np.quantile(pts[:, 2], 0.05))
            pts = pts[pts[:, 2] <= z_floor_est + self._max_ceiling_above_floor]
        if len(pts):
            stamp_pub = (
                self._latest_stamp
                if self._latest_stamp is not None
                else self.get_clock().now().to_msg()
            )
            self._filtered_scan_pub.publish(xyz_to_pc2(pts, self.odom_frame, stamp_pub))
        scan = voxel_downsample(pts, self._voxel_scan)

        if not self._initialized:
            sc = scan.mean(0)
            sz = scan[:, 2]
            self.get_logger().info(
                f"Scan (world frame): {len(scan)} pts, "
                f"centroid=({sc[0]:.2f},{sc[1]:.2f},{sc[2]:.2f}), Z:[{sz.min():.2f},{sz.max():.2f}]"
            )

        t0 = time.monotonic()

        if not self._initialized:
            self._first_localize(scan)
        else:
            self._track(scan)

        self.get_logger().debug(f"ICP took {(time.monotonic()-t0)*1000:.0f} ms")

    def _smooth_tf(self, T_new: np.ndarray) -> np.ndarray:
        """Smooth pose toward T_new. Two modes:

        - Kalman (inauguration pipeline): 4 independent 1-D filters on
          (x, y, z, yaw). Yaw differences are wrapped before filtering.
        - EMA + SLERP (legacy fallback for use_kalman=false).
        """
        if self._use_kalman:
            new_yaw = float(
                Rotation.from_matrix(T_new[:3, :3]).as_euler("xyz", degrees=False)[2]
            )
            # Wrap measurement near the filter's current angle to avoid
            # discontinuities at ±π — feed the measurement closest to state.
            two_pi = 2.0 * np.pi
            delta = (new_yaw - self._kf_yaw.x + np.pi) % two_pi - np.pi
            new_yaw_wrapped = self._kf_yaw.x + delta
            x = self._kf_x.step(T_new[0, 3])
            y = self._kf_y.step(T_new[1, 3])
            z = self._kf_z.step(T_new[2, 3])
            yaw = self._kf_yaw.step(new_yaw_wrapped)
            R_yaw = Rotation.from_euler("z", yaw).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R_yaw
            T[:3, 3] = (x, y, z)
            return T

        alpha = self._tf_smooth_alpha
        t = alpha * T_new[:3, 3] + (1 - alpha) * self._T_published[:3, 3]
        t[2] = T_new[2, 3]
        r_old = Rotation.from_matrix(self._T_published[:3, :3])
        r_new = Rotation.from_matrix(T_new[:3, :3])
        r_smooth = Slerp([0, 1], Rotation.concatenate([r_old, r_new]))(
            [alpha]
        ).as_matrix()[0]
        T = np.eye(4)
        T[:3, :3] = r_smooth
        T[:3, 3] = t
        return T

    def _seed_kalmans_from_T(self, T: np.ndarray) -> None:
        """Reset all 4 Kalman filters to the values from a transform (used
        after a confident cold-start / re-localization)."""
        yaw = float(Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)[2])
        self._kf_x.reset(T[0, 3])
        self._kf_y.reset(T[1, 3])
        self._kf_z.reset(T[2, 3])
        self._kf_yaw.reset(yaw)

    def _rebuild_local_map(self, pose_xy: np.ndarray) -> None:
        """Re-extract a local subset of the full map within `local_map_radius`
        of the given XY pose, downsample, and rebuild kd-trees + normals.
        Called when the robot has moved more than `dis_updatemap` away from
        the centre of the previous subset.
        """
        full = self._tgt_pts_full
        dxy = np.linalg.norm(full[:, :2] - pose_xy[None, :2], axis=1)
        mask = dxy < self._local_map_radius
        subset = full[mask]
        if len(subset) < 200:
            self.get_logger().warning(
                f"Local subset at ({pose_xy[0]:.2f},{pose_xy[1]:.2f}) only has "
                f"{len(subset)} pts — keeping previous map"
            )
            return
        z_mask = (subset[:, 2] >= -0.3) & (subset[:, 2] <= 5.0)
        icp_pts = subset[z_mask]
        self._tgt_pts = voxel_downsample(icp_pts, self._voxel_map_size)
        self._tgt_tree = cKDTree(self._tgt_pts)
        tgt_xy = self._tgt_pts[:, :2]
        self._tgt_tree_2d = cKDTree(tgt_xy)
        self._tgt_pts_2d = tgt_xy
        self._tgt_normals = estimate_normals_safe(
            self._tgt_pts, radius=self._voxel_map_size * 2.5
        )
        self._local_map_centre = pose_xy.copy()
        self.get_logger().info(
            f"Local map rebuilt at xy=({pose_xy[0]:.2f},{pose_xy[1]:.2f}) — "
            f"{len(self._tgt_pts)} ICP pts (radius={self._local_map_radius}m)"
        )

    def _roll_pitch_deg(self, T: np.ndarray) -> tuple[float, float]:
        """Return (|roll|, |pitch|) of T's rotation in degrees. Used to reject
        ICP matches that produce physically-impossible tilts in `map → world`
        (e.g., 180° flips that 3D point-to-plane occasionally finds as a
        local minimum)."""
        rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)

        # roll and pitch should be near zero because both map and scan are
        # gravity-aligned upstream; wrap to [-180, 180] before taking abs.
        def wrap(a: float) -> float:
            return (float(a) + 180.0) % 360.0 - 180.0

        return abs(wrap(rpy[0])), abs(wrap(rpy[1]))

    def _with_floor_z(self, T: np.ndarray) -> np.ndarray:
        """Return T with Z translation set to put floor at Z=0 in map frame."""
        T_out = T.copy()
        T_out[2, 3] = -self._floor_z_world
        return T_out

    def _run_global_icp(self, scan: np.ndarray, hint_T: np.ndarray):
        """Dispatch to global ICP based on the use_3d_icp / use_fpfh_ransac flags.

        Preferred 3D path (matches the inauguration `open3d_loc` pipeline):
        FPFH+RANSAC for a hint-free 6-DOF estimate, then point-to-plane ICP
        refinement at centimetre accuracy. Falls back to the 36-yaw seed
        approach if FPFH/RANSAC is disabled.
        """
        if not self._use_3d_icp:
            return global_icp_2d(
                scan, self._tgt_tree_2d, self._tgt_pts_2d, hint_T, self.icp_max_dist
            )
        if self._use_fpfh_ransac and self._tgt_fpfh is not None:
            scan_normals = estimate_normals_safe(scan, radius=self._voxel_scan * 2.5)
            T_r, fit_r, rmse_r = fpfh_ransac_global(
                scan,
                scan_normals,
                self._tgt_pcd_o3d,
                self._tgt_fpfh,
                feature_radius=self._voxel_map_size * 5.0,
                distance_threshold=self.icp_max_dist * 1.5,
            )
            T_icp, fit_icp, rmse_icp = icp_point_to_plane(
                scan,
                self._tgt_tree,
                self._tgt_pts,
                self._tgt_normals,
                T_r,
                self.icp_max_dist,
            )
            self.get_logger().info(
                f"FPFH+RANSAC → fitness={fit_r:.3f} rmse={rmse_r:.3f}  "
                f"refined → fitness={fit_icp:.3f} rmse={rmse_icp:.3f}"
            )
            return T_icp, fit_icp, rmse_icp
        if self._use_multi_scale_icp:
            # Match inauguration approach: multi-scale point-to-plane from hint.
            # 36-yaw seeds: refine best seed at coarsest voxel only, then run
            # multi-scale ICP cascade on the best to tighten alignment.
            best_T, best_fit, _best_rmse = hint_T.copy(), -1.0, float("inf")
            src_centre = _transform_pts(scan, hint_T).mean(0)
            coarse_v = max(self._multi_scale_voxels)
            src_coarse = voxel_downsample(scan, coarse_v)
            tgt_pcd = o3d.geometry.PointCloud()
            tgt_pcd.points = o3d.utility.Vector3dVector(self._tgt_pts)
            tgt_pcd.normals = o3d.utility.Vector3dVector(self._tgt_normals)
            tgt_coarse_pcd = tgt_pcd.voxel_down_sample(coarse_v)
            tp = np.asarray(tgt_coarse_pcd.points)
            tn = np.asarray(tgt_coarse_pcd.normals)
            tt = cKDTree(tp)
            for yaw_deg in range(0, 360, 5):
                yaw_R = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
                dT = np.eye(4)
                dT[:3, :3] = yaw_R
                dT[:3, 3] = src_centre - yaw_R @ src_centre
                T_init = dT @ hint_T
                T, fit, rmse = icp_point_to_plane(
                    src_coarse,
                    tt,
                    tp,
                    tn,
                    T_init,
                    coarse_v * 2.5,
                    iterations=15,
                )
                if (
                    float(np.linalg.norm(T[:2, 3] - hint_T[:2, 3]))
                    > self._max_init_dist_from_hint
                ):
                    continue
                if fit > best_fit:
                    best_T, best_fit, _best_rmse = T, fit, rmse
            # Refine best seed with fine-scale cascade
            return multi_scale_icp_point_to_plane(
                scan,
                self._tgt_pts,
                self._tgt_normals,
                best_T,
                voxel_sizes=self._multi_scale_voxels,
            )
        return global_icp_point_to_plane(
            scan,
            self._tgt_tree,
            self._tgt_pts,
            self._tgt_normals,
            hint_T,
            self.icp_max_dist,
            max_drift_from_hint=self._max_init_dist_from_hint,
        )

    def _run_track_icp(self, scan: np.ndarray, init_T: np.ndarray):
        """Dispatch to tracking ICP based on use_3d_icp + use_multi_scale_icp flags.

        Tracking uses a fine-only cascade (`track_multi_scale_voxels`, no coarse
        stage): the prior is already close, so coarse voxels only add the risk
        of jumping to a far ambiguous basin and cap the achievable wall
        precision. The coarse cascade stays in `_run_global_icp` for cold-start.
        """
        if not self._use_3d_icp:
            return icp_2d(
                scan[:, :2],
                self._tgt_tree_2d,
                self._tgt_pts_2d,
                init_T,
                self.icp_max_dist,
            )
        if self._use_multi_scale_icp:
            return multi_scale_icp_point_to_plane(
                scan,
                self._tgt_pts,
                self._tgt_normals,
                init_T,
                voxel_sizes=self._track_voxels,
            )
        return icp_point_to_plane(
            scan,
            self._tgt_tree,
            self._tgt_pts,
            self._tgt_normals,
            init_T,
            self.icp_max_dist,
        )

    def _first_localize(self, scan: np.ndarray):
        # Map's floor is already at Z=0 in map frame. World's floor sits at
        # scan_floor_z (≈ -LiDAR_mount_height). Use a 5% quantile for robustness.
        self._floor_z_world = float(np.quantile(scan[:, 2], 0.05))
        self.get_logger().info(f"Floor Z from scan: {self._floor_z_world:.2f}m")

        mode = "3D" if self._use_3d_icp else "2D"
        self.get_logger().info(
            f"Running global {mode} ICP (36 yaw rotations from hint)…"
        )
        T, fitness, rmse = self._run_global_icp(scan, self.T_map_cam)
        T_pub = self._with_floor_z(T)
        roll_deg, pitch_deg = self._roll_pitch_deg(T)
        good_attitude = (
            roll_deg <= self._max_roll_pitch_deg
            and pitch_deg <= self._max_roll_pitch_deg
        )
        dist_from_hint = float(np.linalg.norm(T[:2, 3] - self._init_pose_xy))
        near_hint = dist_from_hint <= self._max_init_dist_from_hint
        if fitness >= self.min_fitness_reloc and good_attitude and near_hint:
            self.T_map_cam = T
            self._seed_kalmans_from_T(T_pub)
            self._T_published = T_pub
            self._initialized = True
            self._settle_ticks = (
                self._settle_ticks_after_lock
            )  # grace before the drift gate clamps
            # Build the local map subset around the first lock so subsequent
            # tracking ICP runs against features near the robot, not the whole map.
            self._rebuild_local_map(T_pub[:2, 3])
            tx, ty = T[0, 3], T[1, 3]
            self.get_logger().info(
                f"Initial localization OK — fitness={fitness:.3f} rmse={rmse:.4f} "
                f"xy=({tx:.2f},{ty:.2f}) floor_z={self._floor_z_world:.2f}"
            )
        else:
            self._T_published = T_pub
            self.get_logger().warning(
                f"Initial localization weak (fitness={fitness:.3f} roll={roll_deg:.1f}deg "
                f"pitch={pitch_deg:.1f}deg dist_from_hint={dist_from_hint:.2f}m). "
                "Publishing hint transform. Provide a better initial_pose if needed."
            )
        self._publish_tf()

    def _track(self, scan: np.ndarray):
        T, fitness, rmse = self._run_track_icp(scan, self.T_map_cam)
        T_pub = self._with_floor_z(T)
        dxy = float(np.linalg.norm(T[:2, 3] - self.T_map_cam[:2, 3]))
        roll_deg, pitch_deg = self._roll_pitch_deg(T)

        # Part 1 — physically-bounded drift gate. `T_map_cam` relates two fixed
        # frames (map, world), so per tick it may only move at Fast-LIO's drift
        # rate. Part 2 widens the budget with robot speed (faster gait → more
        # drift + scan distortion). `max_dpose_per_tick` stays as a hard ceiling.
        drift_budget = (
            self._max_map_drift_rate + self._robot_speed * self._drift_rate_speed_gain
        ) * self._loc_period
        # Post-(re)lock grace: until the pose settles, use the loose ceiling so
        # ICP can converge; only then clamp to the tight physical drift budget.
        if self._settle_ticks > 0:
            self._settle_ticks -= 1
            effective_budget = self.max_dpose_per_tick
        else:
            effective_budget = drift_budget
        within_drift = dxy <= effective_budget

        # Part 4 — a jump past the budget is only accepted if ICP is *decisively*
        # more confident there than at the prior pose. Ambiguous basins (both
        # ~equal fitness) lose to the prior → no flip-flop.
        justified = False
        if not within_drift and dxy <= self.max_dpose_per_tick:
            justified = (
                fitness
                >= self._eval_fitness(scan, self.T_map_cam) + self._jump_fitness_margin
            )

        accept = (
            fitness >= self.min_fitness
            and rmse <= self.max_rmse
            and dxy <= self.max_dpose_per_tick
            and (within_drift or justified)
            and roll_deg <= self._max_roll_pitch_deg
            and pitch_deg <= self._max_roll_pitch_deg
        )

        if accept:
            self.T_map_cam = T
            # Confidence gate: only update the Kalman filter on high-confidence
            # measurements (matches open3d_loc's confidence_loc_th).
            if fitness >= self._confidence_th:
                self._T_published = self._smooth_tf(T_pub)
            else:
                # Hold previous smoothed pose; keep T_map_cam (the ICP prior)
                # so the next tick still iterates from the latest match.
                pass
            self._lost_count = 0
            # Rebuild the local map subset if we've drifted past dis_updatemap.
            pose_xy = T_pub[:2, 3]
            if (
                self._local_map_centre is None
                or np.linalg.norm(pose_xy - self._local_map_centre)
                > self._dis_updatemap
            ):
                self._rebuild_local_map(pose_xy)
            tx, ty = T[0, 3], T[1, 3]
            if not within_drift:
                jump_tag = " [justified jump]"
            elif self._settle_ticks > 0:
                jump_tag = " [settling]"
            else:
                jump_tag = ""
            self.get_logger().info(
                f"ICP track — fitness={fitness:.3f} rmse={rmse:.4f} Δxy={dxy:.2f}m "
                f"(budget={effective_budget:.2f}m speed={self._robot_speed:.2f}m/s) "
                f"xy=({tx:.2f},{ty:.2f}){jump_tag}"
            )
        else:
            self._lost_count += 1
            reasons = []
            if fitness < self.min_fitness:
                reasons.append(f"fitness {fitness:.3f}<{self.min_fitness}")
            if rmse > self.max_rmse:
                reasons.append(f"rmse {rmse:.3f}>{self.max_rmse}")
            if dxy > self.max_dpose_per_tick:
                reasons.append(f"Δxy {dxy:.2f}>{self.max_dpose_per_tick}m ceiling")
            elif not within_drift and not justified:
                reasons.append(
                    f"Δxy {dxy:.2f}>{effective_budget:.2f}m drift-budget "
                    f"(speed={self._robot_speed:.2f}m/s), jump not fitness-justified"
                )
            if roll_deg > self._max_roll_pitch_deg:
                reasons.append(f"roll {roll_deg:.1f}>{self._max_roll_pitch_deg}deg")
            if pitch_deg > self._max_roll_pitch_deg:
                reasons.append(f"pitch {pitch_deg:.1f}>{self._max_roll_pitch_deg}deg")
            self.get_logger().warning(
                f"ICP track rejected ({'; '.join(reasons)}); lost={self._lost_count} — keeping last pose"
            )
            if self._lost_count >= self.relocalize_after_n_losses:
                mode = "3D" if self._use_3d_icp else "2D"
                self.get_logger().info(
                    f"Lost {self._lost_count} ticks — running global {mode} re-localization"
                )
                T_g, fit_g, rmse_g = self._run_global_icp(scan, self.T_map_cam)
                roll_g, pitch_g = self._roll_pitch_deg(T_g)
                good_attitude_g = (
                    roll_g <= self._max_roll_pitch_deg
                    and pitch_g <= self._max_roll_pitch_deg
                )
                drift_g = float(np.linalg.norm(T_g[:2, 3] - self.T_map_cam[:2, 3]))
                near_prior = drift_g <= self._max_recovery_drift_from_prior
                if (
                    fit_g >= self.min_fitness_reloc
                    and rmse_g <= self.max_rmse
                    and good_attitude_g
                    and near_prior
                ):
                    self.T_map_cam = T_g
                    T_g_pub = self._with_floor_z(T_g)
                    self._seed_kalmans_from_T(T_g_pub)
                    self._T_published = T_g_pub
                    self._lost_count = 0
                    self._settle_ticks = (
                        self._settle_ticks_after_lock
                    )  # grace before the drift gate clamps
                    self._rebuild_local_map(T_g_pub[:2, 3])
                    tx, ty = T_g[0, 3], T_g[1, 3]
                    self.get_logger().info(
                        f"Re-localized — fitness={fit_g:.3f} rmse={rmse_g:.4f} xy=({tx:.2f},{ty:.2f})"
                    )
                else:
                    self.get_logger().warning(
                        f"Re-localization weak (fitness={fit_g:.3f} rmse={rmse_g:.4f} "
                        f"roll={roll_g:.1f}deg pitch={pitch_g:.1f}deg drift={drift_g:.2f}m) — keeping last pose"
                    )
        self._publish_tf()

    def _publish_map_cloud(self):
        # Map cloud lives in `map` frame: it was gravity-aligned at load time, so
        # Foxglove with fixed_frame=map renders the floor horizontally with no TF chain.
        stamp = (
            self._latest_stamp
            if self._latest_stamp is not None
            else self.get_clock().now().to_msg()
        )
        msg = xyz_to_pc2(self._display_pts, self.map_frame, stamp)
        self._map_pub.publish(msg)

    def _publish_tf(self):
        # Use cloud stamp to stay in Fast-LIO time domain (boot time, not Unix wall clock)
        stamp = (
            self._latest_stamp
            if self._latest_stamp is not None
            else self.get_clock().now().to_msg()
        )
        ts = matrix_to_ros_tf(
            self._T_published,
            stamp,
            self.map_frame,
            self.odom_frame,
        )
        self._tf_br.sendTransform(ts)


# --------------------------------------------------------------------------- #


def main():
    rclpy.init()
    node = LocalizerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
