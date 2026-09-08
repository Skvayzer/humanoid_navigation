#!/usr/bin/env python3
from __future__ import annotations

"""Floor-RANSAC visualization alignment for a Fast-LIO registered cloud.

Fast-LIO derives the ``world -> camera_init`` rotation from the IMU gravity
estimate alone. That estimate keeps a small residual tilt (typically ~1 deg),
so the gravity-aligned scan does not sit perfectly level against the
floor-RANSAC-aligned map inside the localizer. The mismatch is enough to stop
3D ICP from converging.

This tool fits the floor plane of the live registered scan and measures how far
that floor normal is from +Z.  It deliberately does *not* replace Fast-LIO's
``world -> camera_init`` transform: doing that would create two TF authorities
for the same child frame.  Instead it publishes a separate corrected
``world -> camera_init_floor`` transform and relabels a copy of the registered
cloud into that corrected frame for visualization.

Usage::

    # auto: read Fast-LIO's transform, fit once, publish a separate view
    python3 ransac_gravity_override.py --publish

    # explicit quat, measure only (prints corrected quaternion + residual)
    python3 ransac_gravity_override.py --imu_quat=-0.003,-0.669,0.0,0.743

If ``--imu_quat`` is omitted the script looks up the ``world -> camera_init``
transform Fast-LIO publishes on ``/tf_static`` and uses that — so it needs no
manual quaternion and can run unattended from ``run_localizer.sh``.
``imu_quat`` is the (qx, qy, qz, qw) Fast-LIO logs as
``Gravity-aligned world frame published: q=[...]``.
"""

import argparse
import sys

import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener


def rotation_as_matrix(rotation: Rotation) -> np.ndarray:
    """Return a rotation matrix on both old and new SciPy releases."""
    if hasattr(rotation, "as_matrix"):
        return rotation.as_matrix()
    return rotation.as_dcm()


def rotation_from_matrix(matrix: np.ndarray) -> Rotation:
    """Create a Rotation on SciPy 1.3 (Foxy image) or newer SciPy."""
    if hasattr(Rotation, "from_matrix"):
        return Rotation.from_matrix(matrix)
    return Rotation.from_dcm(matrix)


def lookup_imu_quat(timeout_s: float = 10.0) -> np.ndarray:
    """Read Fast-LIO's ``world -> camera_init`` rotation from TF.

    Args:
        timeout_s: How long to spin waiting for the transform to appear.

    Returns:
        (qx, qy, qz, qw) of the ``world -> camera_init`` transform.
    """
    node = Node("gravity_override_tf_probe")
    buf = Buffer()
    TransformListener(buf, node)
    deadline = node.get_clock().now().nanoseconds + int(timeout_s * 1e9)
    quat = None
    while rclpy.ok() and node.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        try:
            tf = buf.lookup_transform("world", "camera_init", rclpy.time.Time())
            r = tf.transform.rotation
            quat = np.array([r.x, r.y, r.z, r.w], dtype=np.float64)
            break
        except Exception:
            continue
    node.destroy_node()
    if quat is None:
        sys.exit("Timed out waiting for world->camera_init TF from Fast-LIO")
    return quat


def pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract XYZ from a PointCloud2 message as an (N, 3) float64 array.

    Args:
        msg: Incoming PointCloud2.

    Returns:
        Finite XYZ points as float64.
    """
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


def fit_floor_normal(
    pts: np.ndarray,
    z_quantile: float = 0.25,
    dist_thresh: float = 0.05,
) -> tuple[np.ndarray, float, float]:
    """RANSAC plane fit on the lowest ``z_quantile`` fraction of points.

    Args:
        pts: (N, 3) point cloud.
        z_quantile: Fraction of lowest-Z points treated as floor candidates.
        dist_thresh: RANSAC inlier distance threshold in metres.

    Returns:
        ``(unit_normal, plane_offset_d, candidate_inlier_ratio)`` with the
        normal forced to positive Z.
    """
    if len(pts) < 200:
        return np.array([0.0, 0.0, 1.0]), 0.0, 0.0
    z_thresh = np.quantile(pts[:, 2], z_quantile)
    cands = pts[pts[:, 2] <= z_thresh]
    if len(cands) < 200:
        return np.array([0.0, 0.0, 1.0]), 0.0, 0.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(cands, dtype=np.float64)
    )
    plane, inliers = pcd.segment_plane(
        distance_threshold=dist_thresh, ransac_n=3, num_iterations=500
    )
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    if n[2] < 0.0:
        n = -n
        d = -d
    nrm = float(np.linalg.norm(n))
    return n / nrm, float(d) / nrm, float(len(inliers)) / float(len(cands))


def rotation_to_align(n_from: np.ndarray, n_to: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation taking unit vector ``n_from`` onto ``n_to`` (Rodrigues).

    Args:
        n_from: Source unit vector.
        n_to: Target unit vector.

    Returns:
        3x3 rotation matrix.
    """
    v = np.cross(n_from, n_to)
    s = float(np.linalg.norm(v))
    c = float(np.dot(n_from, n_to))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64
    )
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


class GravityOverride(Node):
    """Compute and publish an isolated floor-aligned visualization frame.

    The registered cloud is published by Fast-LIO in the ``camera_init`` frame.
    Rotating it by the IMU quaternion lifts it into ``world``; the residual tilt
    of its floor plane is exactly the error in Fast-LIO's gravity estimate.
    """

    # Hard floor on accumulated points: fit_floor_normal needs >=200 floor
    # candidates (lowest 25%), so the cloud must carry at least ~800 points.
    # /cloud_registered is decimated (~600/scan) so several scans are pooled.
    MIN_POINTS = 2500
    MAX_SCANS = 60

    def __init__(
        self,
        imu_quat: np.ndarray,
        min_points: int,
        publish: bool,
        input_topic: str,
        output_topic: str,
        output_frame: str,
        max_correction_deg: float,
        min_inlier_ratio: float,
        flip_gravity: bool,
    ) -> None:
        """Initialise the node.

        Args:
            imu_quat: (qx, qy, qz, qw) of Fast-LIO's ``world -> camera_init`` TF.
            min_points: Minimum pooled points before the floor fit runs.
            publish: When True, broadcast the corrected static TF and keep spinning.
        """
        super().__init__("gravity_override")
        self._r_imu = rotation_as_matrix(Rotation.from_quat(imu_quat))
        if flip_gravity:
            # The G1's Livox IMU is mounted with its vertical axis inverted
            # relative to the desired visualization world. A proper 180-degree
            # rotation about world X flips Y/Z while preserving forward X.
            self._r_imu = np.diag([1.0, -1.0, -1.0]) @ self._r_imu
        self._min_points = max(min_points, self.MIN_POINTS)
        self._publish = publish
        self._input_topic = input_topic
        self._output_topic = output_topic
        self._output_frame = output_frame
        self._max_correction_deg = max_correction_deg
        self._min_inlier_ratio = min_inlier_ratio
        self._scans: list[np.ndarray] = []
        self._n_seen = 0
        self._done = False
        self._active = False

        if flip_gravity:
            self.get_logger().warn(
                "Applying G1 visualization gravity flip: 180deg about world X. "
                "Fast-LIO estimator TF remains unchanged."
            )

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(
            PointCloud2, self._input_topic, self._on_cloud, qos
        )
        self._cloud_pub = self.create_publisher(PointCloud2, self._output_topic, qos)
        self._static_tf = StaticTransformBroadcaster(self)
        self.get_logger().info(
            f"Pooling {self._input_topic} until >={self._min_points} pts for floor RANSAC..."
        )

    def _on_cloud(self, msg: PointCloud2) -> None:
        """Pool scans until enough points are gathered, then compute the correction."""
        if self._active:
            msg.header.frame_id = self._output_frame
            self._cloud_pub.publish(msg)
            return
        if self._done:
            return
        pts = pc2_to_xyz(msg)
        if len(pts):
            self._scans.append(pts)
        self._n_seen += 1
        total = sum(len(s) for s in self._scans)
        if total >= self._min_points or self._n_seen >= self.MAX_SCANS:
            self._done = True
            self._compute()
            if self._active:
                msg.header.frame_id = self._output_frame
                self._cloud_pub.publish(msg)

    def _compute(self) -> None:
        """Fit the floor, derive the corrected quaternion, log and optionally publish."""
        scan = np.vstack(self._scans) if self._scans else np.empty((0, 3))
        world_pts = (self._r_imu @ scan.T).T

        # Replicate fit_floor_normal's candidate gate so a silent fallback
        # ([0,0,1], residual 0.00) can never be mistaken for a real fit.
        n_cands = 0
        if len(world_pts) >= 200:
            z_thresh = np.quantile(world_pts[:, 2], 0.25)
            n_cands = int((world_pts[:, 2] <= z_thresh).sum())
        if n_cands < 200:
            self.get_logger().error(
                f"Floor RANSAC starved: {len(world_pts)} pts, {n_cands} floor candidates "
                f"(<200). /cloud_registered too sparse — NOT publishing. "
                f"Raise --min_points or enable dense_publish_en."
            )
            rclpy.shutdown()
            sys.exit(2)

        n_floor, floor_d, inlier_ratio = fit_floor_normal(world_pts)
        residual_deg = float(np.degrees(np.arccos(np.clip(n_floor[2], -1.0, 1.0))))

        self.get_logger().info(
            f"floor candidate: normal=[{n_floor[0]:.4f},{n_floor[1]:.4f},"
            f"{n_floor[2]:.4f}] correction={residual_deg:.2f}deg "
            f"inlier_ratio={inlier_ratio:.3f} plane_d={floor_d:.3f}m"
        )

        if inlier_ratio < self._min_inlier_ratio:
            self.get_logger().error(
                f"Floor fit rejected: candidate inlier ratio {inlier_ratio:.3f} is below "
                f"{self._min_inlier_ratio:.3f}; retrying with fresh scans."
            )
            self._scans.clear()
            self._n_seen = 0
            self._done = False
            return
        if residual_deg > self._max_correction_deg:
            self.get_logger().error(
                f"Floor fit rejected: requested correction {residual_deg:.2f}deg exceeds "
                f"the {self._max_correction_deg:.2f}deg safety limit; retrying."
            )
            self._scans.clear()
            self._n_seen = 0
            self._done = False
            return

        r_correct = rotation_to_align(n_floor, np.array([0.0, 0.0, 1.0]))
        r_corrected = r_correct @ self._r_imu
        qx, qy, qz, qw = rotation_from_matrix(r_corrected).as_quat()

        self.get_logger().info(
            f"floor normal (world) = [{n_floor[0]:.4f},{n_floor[1]:.4f},{n_floor[2]:.4f}] "
            f"residual {residual_deg:.2f}deg  pts={len(scan)} "
            f"candidate_inliers={inlier_ratio:.3f}"
        )
        self.get_logger().info(f"QUAT: {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

        if not self._publish:
            rclpy.shutdown()
            return

        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "world"
        tf.child_frame_id = self._output_frame
        # After r_correct aligns the plane normal with +Z, its equation is
        # z + floor_d = 0. Translating the child origin by +floor_d in world Z
        # places that detected floor on the Foxglove Z=0 ground grid.
        tf.transform.translation.z = float(floor_d)
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)
        tf.transform.rotation.w = float(qw)
        self._static_tf.sendTransform(tf)
        self._active = True
        self._scans.clear()
        self.get_logger().info(
            f"Published isolated world -> {self._output_frame} static TF; relaying "
            f"{self._input_topic} -> {self._output_topic}; floor shifted by "
            f"{floor_d:.3f}m to world Z=0. Fast-LIO TF is unchanged."
        )


def main(
    imu_quat: str = "",
    min_points: int = 2500,
    publish: bool = False,
    input_topic: str = "/g1_slam/cloud_registered",
    output_topic: str = "/g1_viz/cloud_registered_floor",
    output_frame: str = "camera_init_floor",
    tf_timeout: float = 30.0,
    max_correction_deg: float = 15.0,
    min_inlier_ratio: float = 0.20,
    flip_gravity: bool = False,
) -> None:
    """Compute the floor-RANSAC-corrected gravity quaternion.

    Args:
        imu_quat: Comma-separated ``qx,qy,qz,qw`` from Fast-LIO's gravity log
            line. If empty, the ``world -> camera_init`` TF is read instead.
        min_points: Minimum pooled ``/cloud_registered`` points before the floor
            fit runs. ``/cloud_registered`` is decimated (~600 pts/scan) so a
            single scan starves the RANSAC; several are pooled.
        publish: When True, broadcast the corrected ``world -> camera_init`` static
            TF and keep the node alive; otherwise print the quaternion and exit.
    """
    rclpy.init()
    if imu_quat:
        quat = np.array([float(v) for v in imu_quat.split(",")], dtype=np.float64)
        if quat.shape != (4,):
            sys.exit("imu_quat must be 'qx,qy,qz,qw'")
    else:
        quat = lookup_imu_quat(tf_timeout)
        print(f"world->camera_init from TF: {quat.tolist()}")

    node = GravityOverride(
        quat,
        min_points,
        publish,
        input_topic,
        output_topic,
        output_frame,
        max_correction_deg,
        min_inlier_ratio,
        flip_gravity,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imu_quat",
        default="",
        help="qx,qy,qz,qw of world->camera_init (omit to read from TF)",
    )
    parser.add_argument(
        "--min_points", type=int, default=2500, help="min pooled points before fit"
    )
    parser.add_argument(
        "--publish", action="store_true", help="broadcast corrected static TF"
    )
    parser.add_argument("--input_topic", default="/g1_slam/cloud_registered")
    parser.add_argument("--output_topic", default="/g1_viz/cloud_registered_floor")
    parser.add_argument("--output_frame", default="camera_init_floor")
    parser.add_argument("--tf_timeout", type=float, default=30.0)
    parser.add_argument("--max_correction_deg", type=float, default=15.0)
    parser.add_argument("--min_inlier_ratio", type=float, default=0.20)
    parser.add_argument(
        "--flip_gravity",
        action="store_true",
        help="flip visualization gravity 180 degrees about world X before RANSAC",
    )
    args = parser.parse_args()
    main(
        imu_quat=args.imu_quat,
        min_points=args.min_points,
        publish=args.publish,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        output_frame=args.output_frame,
        tf_timeout=args.tf_timeout,
        max_correction_deg=args.max_correction_deg,
        min_inlier_ratio=args.min_inlier_ratio,
        flip_gravity=args.flip_gravity,
    )
