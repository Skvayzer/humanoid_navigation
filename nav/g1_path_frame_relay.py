#!/usr/bin/env python3
"""Normalize Foxy Navfn Path pose frames for Foxglove visualization only."""

import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node


class PathFrameRelay(Node):
    def __init__(self):
        super().__init__("g1_path_frame_relay", namespace="g1_nav")
        self.publisher = self.create_publisher(Path, "plan", 10)
        self.create_subscription(Path, "plan_raw", self.on_path, 10)
        self.get_logger().info(
            "visualization relay: /g1_nav/plan_raw -> /g1_nav/plan"
        )

    def on_path(self, message):
        frame = message.header.frame_id.lstrip("/") or "map"
        message.header.frame_id = frame
        for pose in message.poses:
            pose.header.frame_id = frame
        self.publisher.publish(message)


def main():
    rclpy.init()
    node = PathFrameRelay()
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
