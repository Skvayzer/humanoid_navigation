#!/usr/bin/env python3
"""Measured map->body pose for a G1 that is standing still (no SLAM).

CAT needs a floor-aligned `map` frame and the IMU/body pose at sensor time.
With the robot stationary this pose is constant and can be MEASURED:
  * roll/pitch from the IMU gravity direction (specific force at rest is "up"),
  * height from a least-squares floor-plane fit to the LiDAR returns, using the
    same lidar->body extrinsics as FAST-LIO/CAT (diag(1,-1,-1), T),
  * yaw fixed so the projected body X axis is map +X (map origin under the IMU).
It is published on /tf at 50 Hz, stamped with the vendor IMU clock (the clock of
the LiDAR timestamps). A motion guard STOPS publishing for good if the robot
rotates or the gravity direction drifts; CAT then goes stale instead of showing
a wrong map. Restart this node after the robot is still again.
Nothing here commands the robot.
"""
import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage
from livox_ros_driver2.msg import CustomMsg

LIDAR_TO_BODY_R = np.diag([1.0, -1.0, -1.0])           # same as FAST-LIO / CAT
LIDAR_TO_BODY_T = np.array([-0.011, -0.02329, 0.04412])


def quat_from_rows(rows):
    """XYZW quaternion of the rotation whose matrix (map<-body) is `rows`."""
    m = rows
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0, 0.0, 0.0, (m[k, j] - m[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
    q = np.asarray(q)
    return q / np.linalg.norm(q)


class StationaryPose(Node):
    def __init__(self):
        super().__init__("stationary_pose", namespace="g1_cat_bridge")
        p = self.declare_parameter
        self.settle_s = p("settle_seconds", 3.0).value
        self.max_gyro = p("max_gyro_rad_s", 0.25).value
        self.max_tilt_change = math.radians(p("max_gravity_change_deg", 2.0).value)
        self.rate_hz = p("rate_hz", 50.0).value
        self.imu = []
        self.clouds = []
        self.rows = None
        self.height = None
        self.up0 = None
        self.up_ema = None
        self.stopped = False
        self.last_pub = 0.0
        self.tf_pub = self.create_publisher(TFMessage, "/tf", 10)
        self.create_subscription(Imu, "/livox/imu", self.on_imu, qos_profile_sensor_data)
        self.cloud_sub = self.create_subscription(CustomMsg, "/livox/lidar", self.on_cloud, qos_profile_sensor_data)
        self.start = time.monotonic()
        self.get_logger().info("Measuring stationary pose; keep the robot still ...")

    def on_cloud(self, msg):
        if self.rows is None and len(self.clouds) < 30:
            self.clouds.append(np.array([(q.x, q.y, q.z) for q in msg.points], dtype=np.float64))

    def on_imu(self, msg):
        f = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        w = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        if self.rows is None:
            self.imu.append((f, w))
            if time.monotonic() - self.start >= self.settle_s and len(self.clouds) >= 10 and len(self.imu) >= 200:
                self.calibrate()
            return
        if self.stopped:
            return
        up = f / max(np.linalg.norm(f), 1e-6)
        self.up_ema = 0.98 * self.up_ema + 0.02 * up
        change = math.acos(float(np.clip(self.up_ema @ self.up0 / np.linalg.norm(self.up_ema), -1, 1)))
        if np.linalg.norm(w) > self.max_gyro or change > self.max_tilt_change:
            self.stopped = True
            self.get_logger().error(
                f"ROBOT MOVED (|gyro|={np.linalg.norm(w):.2f} rad/s, gravity change={math.degrees(change):.1f} deg). "
                "Stopped publishing map->body; CAT output will go stale. Restart when still.")
            return
        now = time.monotonic()
        if now - self.last_pub >= 1.0 / self.rate_hz:
            self.last_pub = now
            self.publish(msg.header.stamp)

    def calibrate(self):
        f = np.array([a for a, _ in self.imu])
        w = np.array([b for _, b in self.imu])
        if np.linalg.norm(w, axis=1).max() > self.max_gyro or f.std(0).max() > 0.05:
            self.get_logger().warn("Robot not still during calibration; retrying ...")
            self.imu.clear(); self.clouds.clear(); self.start = time.monotonic()
            return
        up = f.mean(0) / np.linalg.norm(f.mean(0))
        x = np.array([1.0, 0.0, 0.0]) - up * up[0]
        x /= np.linalg.norm(x)
        rows = np.vstack([x, np.cross(up, x), up])        # map axes expressed in body
        body = np.concatenate(self.clouds) @ LIDAR_TO_BODY_R.T + LIDAR_TO_BODY_T
        pts = body @ rows.T
        r = np.hypot(pts[:, 0], pts[:, 1])
        sel = (r > 0.6) & (r < 6.0) & (pts[:, 2] < -0.3)
        if sel.sum() < 2000:
            self.get_logger().error("Too few returns below the sensor to find the floor; retrying ...")
            self.imu.clear(); self.clouds.clear(); self.start = time.monotonic()
            return
        hist, edges = np.histogram(pts[sel, 2], bins=np.arange(-3.0, -0.3, 0.02))
        z0 = edges[int(np.argmax(hist))] + 0.01
        band = sel & (np.abs(pts[:, 2] - z0) < 0.05)
        a = np.column_stack([pts[band, 0], pts[band, 1], np.ones(band.sum())])
        coef, *_ = np.linalg.lstsq(a, pts[band, 2], rcond=None)
        rms = float(np.std(pts[band, 2] - a @ coef))
        slope = math.degrees(math.atan(math.hypot(coef[0], coef[1])))
        if band.sum() < 1000 or rms > 0.05 or slope > 3.0:
            self.get_logger().error(f"Floor fit rejected (n={band.sum()}, rms={rms:.3f} m, slope={slope:.1f} deg); retrying ...")
            self.imu.clear(); self.clouds.clear(); self.start = time.monotonic()
            return
        self.rows, self.height = rows, -float(coef[2])
        self.up0 = self.up_ema = up
        self.quat = quat_from_rows(rows)
        self.destroy_subscription(self.cloud_sub)
        tilt = math.degrees(math.acos(abs(up[2])))
        self.get_logger().info(
            f"Pose fixed: IMU height above floor {self.height:.3f} m, tilt {tilt:.2f} deg, "
            f"floor fit n={band.sum()} rms={rms*100:.1f} cm slope={slope:.2f} deg, "
            f"quat_xyzw={np.round(self.quat, 6).tolist()}. Publishing map->body at {self.rate_hz:.0f} Hz.")

    def publish(self, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = "body"
        t.transform.translation.z = self.height
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = map(float, self.quat)
        self.tf_pub.publish(TFMessage(transforms=[t]))


def main():
    rclpy.init(args=sys.argv)
    node = StationaryPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
