#!/usr/bin/env python3
"""Records per-scan stamp, PC2 arrival time and per-point time stats of the vendor cloud (plus IMU stamps).

    ros2 run g1_sensor_restamp record_scan_info.py 60 > scan_info.csv

Rows: "I,<stamp_ns>" and "C,<stamp_ns>,<arrival_ns>,<n_points>,<t_first>,<t_last>,<t_min>,<t_max>" (point
times in ns). Used to study how the vendor's cloud processing affects the IMU stream: the IMU stalls
end at cloud stamps, so the per-scan timing shows when and for how long the IMU path is blocked.
"""
import sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, Imu

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
PT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4"), ("ring", "<u2"), ("t", "<f4")])


def ns(m):
    return m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec


class Rec(Node):
    def __init__(self):
        super().__init__("rv_scan_info")
        self.rows = []
        self.create_subscription(Imu, "/utlidar/imu_livox_mid360", self.on_imu,
                                 QoSProfile(depth=500, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(PointCloud2, "/utlidar/cloud_livox_mid360", self.on_cloud,
                                 QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE))

    def on_imu(self, m):
        self.rows.append(f"I,{ns(m)}")

    def on_cloud(self, m):
        arrival = time.time_ns()
        p = np.frombuffer(bytes(m.data), dtype=PT) if m.point_step == 22 else None
        t = p["t"]
        # first/last point in storage order, plus min/max
        self.rows.append(f"C,{ns(m)},{arrival},{len(t)},{t[0]:.1f},{t[-1]:.1f},{t.min():.1f},{t.max():.1f}")


rclpy.init()
n = Rec()
end = time.time() + DUR
while time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.01)
sys.stdout.write("\n".join(n.rows) + "\n")
n.destroy_node()
rclpy.shutdown()
