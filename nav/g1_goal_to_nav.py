#!/usr/bin/env python3
"""Convert Foxglove goal clicks into Nav2 NavigateToPose actions.

This node does not publish velocity. It only sends goals to bt_navigator.
The controller and the G1 velocity bridge own motion once the user clicks.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

GOAL_DEDUP_S = float(os.environ.get("G1_NAV_GOAL_DEDUP_S", "5.0"))
GOAL_FRESH_S = float(os.environ.get("G1_NAV_GOAL_FRESH_S", "15.0"))
# Ignore cancel_navigation=True signals arriving within this many seconds of
# sending a goal. Filters stale TRANSIENT_LOCAL replays and safety-system signals
# that fire immediately after a failed nav session. G1_NAV_CANCEL_GRACE_S=0 disables.
CANCEL_GRACE_S = float(os.environ.get("G1_NAV_CANCEL_GRACE_S", "5.0"))
FINAL_SPIN_TO_GOAL_YAW = os.environ.get(
    "G1_NAV_FINAL_SPIN_TO_GOAL_YAW", "true"
).strip().lower() in {"1", "true", "yes", "on"}
FINAL_YAW_TOLERANCE_RAD = float(
    os.environ.get("G1_NAV_FINAL_YAW_TOLERANCE_RAD", "0.10")
)
FINAL_SPIN_TIMEOUT_S = float(os.environ.get("G1_NAV_FINAL_SPIN_TIMEOUT_S", "30.0"))
FINAL_SPIN_ACTION = os.environ.get("G1_NAV_FINAL_SPIN_ACTION", "/g1_nav/spin")
FINAL_HEADING_FRAME = os.environ.get("G1_NAV_FINAL_HEADING_FRAME", "g1_nav_base")
FINAL_HEADING_FALLBACK_FRAME = os.environ.get(
    "G1_NAV_FINAL_HEADING_FALLBACK_FRAME", "body_level"
)

STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


def _quat_to_mat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    m = np.eye(3)
    m[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    m[0, 1] = 2 * (qx * qy - qz * qw)
    m[0, 2] = 2 * (qx * qz + qy * qw)
    m[1, 0] = 2 * (qx * qy + qz * qw)
    m[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    m[1, 2] = 2 * (qy * qz - qx * qw)
    m[2, 0] = 2 * (qx * qz - qy * qw)
    m[2, 1] = 2 * (qy * qz + qx * qw)
    m[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return m


def _rot_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return x / norm, y / norm, z / norm, w / norm


def _transform_to_mat(tf) -> np.ndarray:
    tr = tf.transform.translation
    ro = tf.transform.rotation
    m = np.eye(4)
    m[:3, :3] = _quat_to_mat(ro.x, ro.y, ro.z, ro.w)
    m[:3, 3] = [tr.x, tr.y, tr.z]
    return m


class GoalToNav(Node):
    """Send every Foxglove click as a single active NavigateToPose goal."""

    def __init__(self) -> None:
        super().__init__("g1_goal_to_nav")
        self._client = ActionClient(self, NavigateToPose, "/g1_nav/navigate_to_pose")
        self._spin_client = ActionClient(self, Spin, FINAL_SPIN_ACTION)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._map: OccupancyGrid | None = None
        self._active_goal = None
        self._active_goal_pose: PoseStamped | None = None
        self._active_spin = None
        self._active_spin_goal_pose: PoseStamped | None = None
        self._pending_goal: tuple[PoseStamped, str, float] | None = None
        self._last_sent_goal_pose: PoseStamped | None = None
        self._last_sent_goal_time: float = 0.0
        self._cancel_ignore_until: float = 0.0
        self._active_goal_source_stamp: float = 0.0
        self._last_succeeded_source_stamp: float = -1.0
        self._last_feedback_log = 0.0
        self._last_stale_goal_log = 0.0
        self._ignore_goal_yaw = os.environ.get(
            "G1_NAV_IGNORE_GOAL_YAW", "true"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._log_path = Path(
            os.environ.get(
                "G1_NAV2_NAV_DEBUG_LOG",
                "/data/logs/nav/g1_nav2_nav_debug.log",
            )
        )

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, "/g1_nav/map", self._on_map, map_qos)

        self.create_subscription(
            PoseStamped,
            "/g1_nav/goal_pose",
            lambda msg: self._on_goal(msg, "/g1_nav/goal_pose"),
            10,
        )
        self.create_subscription(
            PoseStamped,
            "/g1_nav/move_base_simple/goal",
            lambda msg: self._on_goal(msg, "/g1_nav/move_base_simple/goal"),
            10,
        )
        self.create_subscription(
            PointStamped, "/g1_nav/clicked_point", self._on_clicked_point, 10
        )
        self.create_subscription(
            Bool, "/g1_nav/cancel_navigation", self._on_cancel_navigation, 10
        )

        self._write_log("g1_goal_to_nav starting")
        self.get_logger().info("Waiting for /g1_nav/navigate_to_pose...")
        self._client.wait_for_server()
        self.get_logger().info(
            "Ready: Foxglove /goal_pose or /move_base_simple/goal will command NavigateToPose"
        )
        self._write_log(
            "navigate_to_pose action server ready; "
            f"final_spin_to_goal_yaw={FINAL_SPIN_TO_GOAL_YAW}, "
            f"spin_action={FINAL_SPIN_ACTION}, "
            f"yaw_tolerance={math.degrees(FINAL_YAW_TOLERANCE_RAD):.1f}deg"
        )

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._map = msg

    def _on_clicked_point(self, msg: PointStamped) -> None:
        goal = PoseStamped()
        goal.header = msg.header
        goal.pose.position.x = msg.point.x
        goal.pose.position.y = msg.point.y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self._on_goal(goal, "/g1_nav/clicked_point")

    def _on_goal(self, goal_pose: PoseStamped, source_topic: str) -> None:
        if not goal_pose.header.frame_id:
            goal_pose.header.frame_id = "map"
        goal_pose.header.frame_id = goal_pose.header.frame_id.lstrip("/")
        source_stamp = (
            goal_pose.header.stamp.sec + goal_pose.header.stamp.nanosec / 1.0e9
        )
        source_yaw = self._yaw_from_quaternion(goal_pose.pose.orientation)
        now = self._now_s()

        # Reject stale TRANSIENT_LOCAL storm replays before TF/map work.
        if source_stamp > 0 and now - source_stamp > GOAL_FRESH_S:
            if now - self._last_stale_goal_log > 5.0:
                self._last_stale_goal_log = now
                message = (
                    f"stale goal ignored from {source_topic}: "
                    f"age={now - source_stamp:.1f}s, source_stamp={source_stamp:.3f}"
                )
                self.get_logger().warning(message)
                self._write_log(message)
            return

        goal_pose = self._goal_in_frame(goal_pose, "map")
        if goal_pose is None:
            return

        # The robot TF tree is currently stamped in the sensor/robot uptime domain,
        # while Foxglove goals arrive with wall-clock stamps. Zero means "latest"
        # to TF consumers and avoids asking Nav2 to transform at an impossible time.
        goal_pose.header.stamp.sec = 0
        goal_pose.header.stamp.nanosec = 0
        if self._ignore_goal_yaw:
            goal_pose.pose.orientation.x = 0.0
            goal_pose.pose.orientation.y = 0.0
            goal_pose.pose.orientation.z = 0.0
            goal_pose.pose.orientation.w = 1.0

        summary = (
            f"goal from {source_topic}: x={goal_pose.pose.position.x:.3f}, "
            f"y={goal_pose.pose.position.y:.3f}, frame={goal_pose.header.frame_id}; "
            f"source_stamp={source_stamp:.3f}, "
            f"source_yaw={math.degrees(source_yaw):.1f}deg, "
            f"ignore_yaw={self._ignore_goal_yaw}; "
            f"goal_cell={self._map_cell_summary(goal_pose)}"
        )
        self.get_logger().info(summary)
        self._write_log(summary)

        # Reject replay of the exact goal that was just reached (same source_stamp).
        if source_stamp > 0 and source_stamp == self._last_succeeded_source_stamp:
            return

        # Suppress active-navigation duplicate (5s window).
        if (
            self._last_sent_goal_pose is not None
            and now - self._last_sent_goal_time < GOAL_DEDUP_S
            and self._same_goal(goal_pose, self._last_sent_goal_pose)
        ):
            return

        if self._active_goal is not None or self._active_spin is not None:
            active_pose = (
                self._active_goal_pose
                if self._active_goal is not None
                else self._active_spin_goal_pose
            )
            if active_pose is not None and self._same_goal(goal_pose, active_pose):
                self.get_logger().info(
                    "Duplicate goal ignored — already navigating there"
                )
                return
            self._pending_goal = (goal_pose, summary, source_stamp)
            self.get_logger().info(
                "Canceling previous navigation goal before sending new one"
            )
            self._write_log("canceling previous navigation goal")
            if self._active_goal is not None:
                cancel_future = self._active_goal.cancel_goal_async()
                cancel_future.add_done_callback(self._on_cancel_done)
            elif self._active_spin is not None:
                cancel_future = self._active_spin.cancel_goal_async()
                cancel_future.add_done_callback(self._on_spin_cancel_done)
            return

        self._send_goal(goal_pose, summary, source_stamp)

    def _goal_in_frame(
        self, pose: PoseStamped, target_frame: str
    ) -> PoseStamped | None:
        if pose.header.frame_id == target_frame:
            return pose
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame,
                pose.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception as exc:  # noqa: BLE001 - log exact TF failure.
            message = (
                f"goal transform failed: {target_frame}<-{pose.header.frame_id}: {exc}"
            )
            self.get_logger().warning(message)
            self._write_log(message)
            return None

        target_T_source = _transform_to_mat(tf)
        p = np.array(
            [
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                1.0,
            ]
        )
        q = pose.pose.orientation
        target_R_goal = target_T_source[:3, :3] @ _quat_to_mat(q.x, q.y, q.z, q.w)
        qx, qy, qz, qw = _rot_to_quat(target_R_goal)

        out = PoseStamped()
        out.header = pose.header
        out.header.frame_id = target_frame
        out.pose.position.x = float((target_T_source @ p)[0])
        out.pose.position.y = float((target_T_source @ p)[1])
        out.pose.position.z = float((target_T_source @ p)[2])
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        return out

    def _on_cancel_done(self, _future) -> None:
        self._active_goal = None
        self._active_goal_pose = None
        pending = self._pending_goal
        self._pending_goal = None
        if pending is not None:
            self._send_goal(pending[0], pending[1], pending[2])

    def _on_spin_cancel_done(self, _future) -> None:
        self._active_spin = None
        self._active_spin_goal_pose = None
        pending = self._pending_goal
        self._pending_goal = None
        if pending is not None:
            self._send_goal(pending[0], pending[1], pending[2])

    def _on_cancel_navigation(self, msg: Bool) -> None:
        if not msg.data:
            self._cancel_ignore_until = 0.0
            return
        now = self._now_s()
        if now < self._cancel_ignore_until:
            self.get_logger().debug(
                f"cancel_navigation ignored: grace period active "
                f"({self._cancel_ignore_until - now:.1f}s remaining)"
            )
            return
        self._pending_goal = None
        if self._active_goal is not None:
            self.get_logger().info("Ana override: canceling navigation goal")
            self._write_log("ana_override: canceling navigation goal")
            cancel_future = self._active_goal.cancel_goal_async()
            cancel_future.add_done_callback(lambda _: None)
            self._active_goal = None
            self._active_goal_pose = None
        if self._active_spin is not None:
            self.get_logger().info("Ana override: canceling final heading spin")
            self._write_log("ana_override: canceling final heading spin")
            cancel_future = self._active_spin.cancel_goal_async()
            cancel_future.add_done_callback(lambda _: None)
            self._active_spin = None
            self._active_spin_goal_pose = None

    def _same_goal(self, a: PoseStamped, b: PoseStamped, tol: float = 0.05) -> bool:
        return (
            abs(a.pose.position.x - b.pose.position.x) < tol
            and abs(a.pose.position.y - b.pose.position.y) < tol
        )

    def _send_goal(
        self, goal_pose: PoseStamped, summary: str, source_stamp: float = 0.0
    ) -> None:
        request = NavigateToPose.Goal()
        request.pose = goal_pose
        self._active_goal_pose = goal_pose
        self._active_goal_source_stamp = source_stamp
        self._last_sent_goal_pose = goal_pose
        self._last_sent_goal_time = self._now_s()
        self._cancel_ignore_until = self._last_sent_goal_time + CANCEL_GRACE_S
        self._last_feedback_log = 0.0
        future = self._client.send_goal_async(
            request, feedback_callback=self._on_feedback
        )
        future.add_done_callback(lambda done: self._on_goal_response(done, summary))

    def _on_feedback(self, feedback_msg) -> None:
        now = self._now_s()
        self._cancel_ignore_until = now + CANCEL_GRACE_S
        if now - self._last_feedback_log < 1.0:
            return
        self._last_feedback_log = now
        feedback = feedback_msg.feedback
        pose = feedback.current_pose.pose.position
        message = (
            "feedback: "
            f"current=({pose.x:.3f},{pose.y:.3f}), "
            f"distance_remaining={feedback.distance_remaining:.3f}, "
            f"recoveries={feedback.number_of_recoveries}"
        )
        self._write_log(message)

    def _on_goal_response(self, future, summary: str) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001 - keep the exact ROS error.
            self.get_logger().error(f"NavigateToPose goal request failed: {exc}")
            self._write_log(f"goal request failed: {exc}; {summary}")
            return

        if not handle.accepted:
            self.get_logger().warning("NavigateToPose rejected goal")
            self._write_log(f"goal rejected; {summary}")
            return

        self._active_goal = handle
        self._cancel_ignore_until = self._now_s() + CANCEL_GRACE_S
        self.get_logger().info("NavigateToPose accepted goal")
        self._write_log(f"goal accepted; {summary}")
        result_future = handle.get_result_async()
        result_future.add_done_callback(lambda done: self._on_result(done, summary))

    def _on_result(self, future, summary: str) -> None:
        finished_goal_pose = self._active_goal_pose
        finished_source_stamp = self._active_goal_source_stamp
        self._active_goal = None
        self._active_goal_pose = None
        try:
            wrapped = future.result()
        except Exception as exc:  # noqa: BLE001 - keep the exact ROS error.
            self.get_logger().error(f"NavigateToPose result failed: {exc}")
            self._write_log(f"result failed: {exc}; {summary}")
            return

        status = STATUS_NAMES.get(wrapped.status, str(wrapped.status))
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            self._last_succeeded_source_stamp = finished_source_stamp
            if self._send_final_spin_if_needed(
                finished_goal_pose, summary, finished_source_stamp
            ):
                self.get_logger().info(
                    "NavigateToPose succeeded; final heading spin started"
                )
                self._write_log(
                    f"result {status}; final heading spin started; {summary}"
                )
                return
            self.get_logger().info("NavigateToPose succeeded")
        else:
            self.get_logger().warning(f"NavigateToPose finished with {status}")
        self._write_log(f"result {status}; {summary}")

    def _send_final_spin_if_needed(
        self,
        goal_pose: PoseStamped | None,
        summary: str,
        source_stamp: float,
    ) -> bool:
        if goal_pose is None or self._ignore_goal_yaw or not FINAL_SPIN_TO_GOAL_YAW:
            return False

        current = self._current_heading_yaw()
        if current is None:
            self._write_log(
                f"final heading skipped: current yaw unavailable; {summary}"
            )
            return False
        current_yaw, heading_frame = current
        target_yaw = self._yaw_from_quaternion(goal_pose.pose.orientation)
        yaw_error = self._angle_diff(target_yaw, current_yaw)
        message = (
            "final heading check: "
            f"target={math.degrees(target_yaw):+.1f}deg, "
            f"current={math.degrees(current_yaw):+.1f}deg frame={heading_frame}, "
            f"error={math.degrees(yaw_error):+.1f}deg"
        )
        self.get_logger().info(message)
        self._write_log(message)

        if abs(yaw_error) <= FINAL_YAW_TOLERANCE_RAD:
            self._write_log("final heading already within tolerance")
            return False

        if not self._spin_client.wait_for_server(timeout_sec=1.0):
            warning = (
                f"final heading skipped: {FINAL_SPIN_ACTION} action server unavailable"
            )
            self.get_logger().warning(warning)
            self._write_log(warning)
            return False

        request = Spin.Goal()
        request.target_yaw = float(yaw_error)
        # Foxy's nav2_msgs/Spin goal contains only target_yaw. Humble added the
        # time_allowance field used by the original X2 adapter.
        self._active_spin_goal_pose = goal_pose
        self._cancel_ignore_until = self._now_s() + CANCEL_GRACE_S
        future = self._spin_client.send_goal_async(
            request, feedback_callback=self._on_spin_feedback
        )
        future.add_done_callback(
            lambda done: self._on_spin_response(done, summary, source_stamp)
        )
        self._write_log(
            f"final heading spin requested: target_yaw={math.degrees(yaw_error):+.1f}deg"
        )
        return True

    def _on_spin_response(self, future, summary: str, source_stamp: float) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001 - keep the exact ROS error.
            self.get_logger().error(f"Final heading spin request failed: {exc}")
            self._write_log(f"final heading spin request failed: {exc}; {summary}")
            self._active_spin_goal_pose = None
            return

        if not handle.accepted:
            self.get_logger().warning("Final heading spin rejected")
            self._write_log(f"final heading spin rejected; {summary}")
            self._active_spin_goal_pose = None
            return

        self._active_spin = handle
        self._active_goal_source_stamp = source_stamp
        self._cancel_ignore_until = self._now_s() + CANCEL_GRACE_S
        self.get_logger().info("Final heading spin accepted")
        self._write_log(f"final heading spin accepted; {summary}")
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda done: self._on_spin_result(done, summary)
        )

    def _on_spin_feedback(self, _feedback_msg) -> None:
        self._cancel_ignore_until = self._now_s() + CANCEL_GRACE_S

    def _on_spin_result(self, future, summary: str) -> None:
        self._active_spin = None
        self._active_spin_goal_pose = None
        try:
            wrapped = future.result()
        except Exception as exc:  # noqa: BLE001 - keep the exact ROS error.
            self.get_logger().error(f"Final heading spin result failed: {exc}")
            self._write_log(f"final heading spin result failed: {exc}; {summary}")
            return

        status = STATUS_NAMES.get(wrapped.status, str(wrapped.status))
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Final heading spin succeeded")
        else:
            self.get_logger().warning(f"Final heading spin finished with {status}")
        self._write_log(f"final heading spin result {status}; {summary}")

        pending = self._pending_goal
        self._pending_goal = None
        if pending is not None:
            self._send_goal(pending[0], pending[1], pending[2])

    def _current_heading_yaw(self) -> tuple[float, str] | None:
        frames = [FINAL_HEADING_FRAME]
        if FINAL_HEADING_FALLBACK_FRAME and FINAL_HEADING_FALLBACK_FRAME not in frames:
            frames.append(FINAL_HEADING_FALLBACK_FRAME)
        for frame in frames:
            try:
                tf = self._tf_buffer.lookup_transform(
                    "map",
                    frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.10),
                )
            except Exception as exc:  # noqa: BLE001 - try fallback, then log.
                self._write_log(f"final heading TF unavailable: map<-{frame}: {exc}")
                continue
            return self._yaw_from_quaternion(tf.transform.rotation), frame
        return None

    def _map_cell_summary(self, pose: PoseStamped) -> str:
        if self._map is None:
            return "no_map_received"
        if pose.header.frame_id != self._map.header.frame_id:
            return f"frame_mismatch(goal={pose.header.frame_id}, map={self._map.header.frame_id})"

        cell = self._world_to_map_cell(pose.pose.position.x, pose.pose.position.y)
        if cell is None:
            return "outside_map"
        col, row, index = cell
        value = self._map.data[index]
        return f"col={col}, row={row}, value={value} ({self._occupancy_name(value)})"

    def _world_to_map_cell(self, x: float, y: float) -> tuple[int, int, int] | None:
        if self._map is None:
            return None
        info = self._map.info
        origin = info.origin
        yaw = self._yaw_from_quaternion(origin.orientation)
        dx = x - origin.position.x
        dy = y - origin.position.y
        map_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        map_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        col = int(math.floor(map_x / info.resolution))
        row = int(math.floor(map_y / info.resolution))
        if col < 0 or row < 0 or col >= info.width or row >= info.height:
            return None
        return col, row, row * info.width + col

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        return math.atan2(math.sin(target - current), math.cos(target - current))

    @staticmethod
    def _occupancy_name(value: int) -> str:
        if value < 0:
            return "unknown"
        if value == 0:
            return "free"
        if value >= 100:
            return "occupied"
        return "cost"

    def _write_log(self, message: str) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
        except OSError as exc:
            self.get_logger().warning(
                f"Could not write debug log {self._log_path}: {exc}"
            )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1.0e9


def main() -> None:
    rclpy.init()
    node = GoalToNav()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
