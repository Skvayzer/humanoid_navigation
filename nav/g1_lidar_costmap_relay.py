#!/usr/bin/env python3
"""Relay FAST-LIO's current body-frame scan for the Nav2 local costmap.

FAST-LIO preserves the Livox acquisition timestamp.  Nav2's TF chain is
published in current ROS wall time, so the relay restamps each complete scan
without changing its points.  Range and height filtering remain in Nav2's
ObstacleLayer, matching the tested X2 navigation pipeline.
"""

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2


INPUT_TOPIC = "/g1_slam/cloud_registered_body"
OUTPUT_TOPIC = "/g1_nav/lidar_costmap"
OUTPUT_FRAME = "body"
LOG_PATH = Path("/data/logs/nav/lidar_costmap_relay.log")


class LidarCostmapRelay(Node):
    def __init__(self):
        super().__init__("g1_lidar_costmap_relay", namespace="g1_nav")
        qos = QoSProfile(depth=2)
        qos.history = QoSHistoryPolicy.KEEP_LAST
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.subscription = self.create_subscription(
            PointCloud2, INPUT_TOPIC, self.cloud_callback, qos
        )
        self.publisher = self.create_publisher(PointCloud2, OUTPUT_TOPIC, qos)
        self.last_log_s = 0.0
        self.log(
            "live lidar relay started: %s -> %s, frame=%s"
            % (INPUT_TOPIC, OUTPUT_TOPIC, OUTPUT_FRAME)
        )

    def cloud_callback(self, message):
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = OUTPUT_FRAME
        self.publisher.publish(message)

        now_s = self.get_clock().now().nanoseconds / 1.0e9
        if now_s - self.last_log_s >= 5.0:
            self.last_log_s = now_s
            self.log(
                "relayed live lidar scan: %d points"
                % (message.width * message.height)
            )

    def log(self, message):
        self.get_logger().info(message)
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
        except OSError as exc:
            self.get_logger().warning("could not write lidar relay log: %s" % exc)


def main():
    rclpy.init()
    node = LidarCostmapRelay()
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
