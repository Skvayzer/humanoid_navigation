#!/usr/bin/env python3
"""Turn a Foxglove goal into a Nav2 path and controller commands.

The controller always publishes on /g1_nav/cmd_vel_preview. That topic can
only move the G1 when the separate host safety gateway is explicitly armed.
"""

from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import OccupancyGrid, Path as PathMsg
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


LOG_PATH = Path("/data/logs/nav/goal_to_plan.log")


class GoalToPlan(Node):
    def __init__(self):
        super().__init__("g1_goal_to_plan", namespace="g1_nav")
        self.client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.controller_client = ActionClient(self, FollowPath, "follow_path")
        self.controller_goal = None
        self.plan_pub = self.create_publisher(PathMsg, "plan", 10)
        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_ready = False
        self.create_subscription(OccupancyGrid, "map", self.on_map, map_qos)
        self.create_subscription(PoseStamped, "goal_pose", self.on_goal, 10)
        self.log(
            "navigation relay started; controller output is on "
            "/g1_nav/cmd_vel_preview (motion gateway must be separately armed)"
        )

    def on_map(self, _message):
        self.map_ready = True

    def on_goal(self, goal):
        if not self.map_ready:
            self.get_logger().warning("goal rejected: occupancy map is not ready")
            return
        frame = goal.header.frame_id.lstrip("/")
        if frame != "map":
            self.get_logger().warning("goal rejected: frame must be map, got '%s'" % frame)
            return
        goal.header.frame_id = "map"
        request = ComputePathToPose.Goal()
        # ROS 2 Foxy names this field "pose". Newer Nav2 releases use
        # "goal" and add "start" / "use_start" fields.
        request.pose = goal
        request.planner_id = "GridBased"
        self.log("requesting path to x=%.3f y=%.3f" % (
            goal.pose.position.x, goal.pose.position.y
        ))
        future = self.client.send_goal_async(request)
        future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            self.log("planner request failed: %s" % exc)
            return
        if not handle.accepted:
            self.log("planner rejected the goal")
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def on_result(self, future):
        try:
            wrapped = future.result()
        except Exception as exc:
            self.log("planner result failed: %s" % exc)
            return
        path = wrapped.result.path
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED and path.poses:
            # Foxy Navfn may leave each PoseStamped frame empty even though the
            # enclosing Path is in map. Foxglove requires them to agree.
            path.header.frame_id = "map"
            for pose in path.poses:
                pose.header.frame_id = "map"
            self.plan_pub.publish(path)
            self.log("published plan with %d poses" % len(path.poses))
            self.start_preview(path)
        else:
            self.log("no path: status=%d poses=%d" % (wrapped.status, len(path.poses)))

    def start_preview(self, path):
        if not self.controller_client.wait_for_server(timeout_sec=0.0):
            self.log("preview not started: FollowPath action is not ready")
            return
        if self.controller_goal is not None:
            self.controller_goal.cancel_goal_async()
            self.controller_goal = None

        request = FollowPath.Goal()
        request.path = path
        request.controller_id = "FollowPath"
        future = self.controller_client.send_goal_async(request)
        future.add_done_callback(self.on_preview_goal_response)

    def on_preview_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            self.log("preview controller request failed: %s" % exc)
            return
        if not handle.accepted:
            self.log("preview controller rejected the path")
            return
        self.controller_goal = handle
        self.log("preview controller accepted the path")
        result_future = handle.get_result_async()
        result_future.add_done_callback(self.on_preview_result)

    def on_preview_result(self, future):
        try:
            wrapped = future.result()
            self.log("preview controller finished with status=%d" % wrapped.status)
        except Exception as exc:
            self.log("preview controller result failed: %s" % exc)
        finally:
            self.controller_goal = None

    def log(self, message):
        self.get_logger().info(message)
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
        except OSError:
            pass


def main():
    rclpy.init()
    node = GoalToPlan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
