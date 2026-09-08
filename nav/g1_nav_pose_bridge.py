#!/usr/bin/env python3
"""Publish a current-time, planar navigation frame from NDT's map->body TF."""

import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


MAP_FRAME = "map"
SOURCE_FRAME = "body"
NAV_BASE_FRAME = "g1_nav_base"
ODOM_TOPIC = "/g1_nav/odom"
POSE_READY_TOPIC = "/g1_nav/pose_ready"
PUBLISH_HZ = 10.0
LOG_PATH = Path("/data/logs/nav/pose_bridge.log")


class PoseBridge(Node):
    def __init__(self):
        super().__init__("g1_nav_pose_bridge", namespace="g1_nav")
        self.buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self.ready_pub = self.create_publisher(Bool, POSE_READY_TOPIC, 10)
        self.last_warning = 0.0
        self.last_pose = None
        self.create_timer(1.0 / PUBLISH_HZ, self.tick)
        self.log("planning-only pose bridge started: map->body => map->g1_nav_base")

    def tick(self):
        try:
            source = self.buffer.lookup_transform(
                MAP_FRAME, SOURCE_FRAME, Time(), timeout=Duration(seconds=0.0)
            )
        except TransformException as exc:
            self.ready_pub.publish(Bool(data=False))
            now = self.get_clock().now().nanoseconds / 1.0e9
            if now - self.last_warning > 2.0:
                self.last_warning = now
                self.get_logger().warning("waiting for map->body: %s" % exc)
            return

        # Unitree G1 body +X is forward. Project that axis into map XY so the
        # Nav2 frame remains planar even when the source pose has roll/pitch.
        q = source.transform.rotation
        forward_x = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        forward_y = 2.0 * (q.x * q.y + q.w * q.z)
        if math.hypot(forward_x, forward_y) < 0.25:
            self.ready_pub.publish(Bool(data=False))
            return
        yaw = math.atan2(forward_y, forward_x)
        half = 0.5 * yaw
        stamp = self.get_clock().now().to_msg()
        x = float(source.transform.translation.x)
        y = float(source.transform.translation.y)

        flat = TransformStamped()
        flat.header.stamp = stamp
        flat.header.frame_id = MAP_FRAME
        flat.child_frame_id = NAV_BASE_FRAME
        flat.transform.translation.x = x
        flat.transform.translation.y = y
        flat.transform.translation.z = 0.0
        flat.transform.rotation.z = math.sin(half)
        flat.transform.rotation.w = math.cos(half)
        self.broadcaster.sendTransform(flat)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = MAP_FRAME
        odom.child_frame_id = NAV_BASE_FRAME
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(half)
        odom.pose.pose.orientation.w = math.cos(half)
        now_s = self.get_clock().now().nanoseconds / 1.0e9
        if self.last_pose is not None:
            last_x, last_y, last_yaw, last_s = self.last_pose
            dt = max(now_s - last_s, 1.0e-3)
            vx_map = (x - last_x) / dt
            vy_map = (y - last_y) / dt
            odom.twist.twist.linear.x = (
                math.cos(yaw) * vx_map + math.sin(yaw) * vy_map
            )
            odom.twist.twist.linear.y = (
                -math.sin(yaw) * vx_map + math.cos(yaw) * vy_map
            )
            yaw_delta = math.atan2(
                math.sin(yaw - last_yaw), math.cos(yaw - last_yaw)
            )
            odom.twist.twist.angular.z = yaw_delta / dt
        self.last_pose = (x, y, yaw, now_s)
        self.odom_pub.publish(odom)
        self.ready_pub.publish(Bool(data=True))

    def log(self, message):
        self.get_logger().info(message)
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
        except OSError as exc:
            self.get_logger().warning("could not write pose bridge log: %s" % exc)


def main():
    rclpy.init()
    node = PoseBridge()
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
