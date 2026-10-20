#!/usr/bin/env python3
"""Records raw samples of the vendor Mid-360 IMU stream for restamp_eval.

    ros2 run g1_sensor_restamp record_stamp_trace.py 60 trace.csv

Rows: "I,<stamp_ns>,ax,ay,az,gx,gy,gz". Values are written with repr() so that repeated samples
stay bit-identical. Subscribes RELIABLE, as the restamp node does by default.
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu

IMU_TOPIC = '/utlidar/imu_livox_mid360'


def stamp_ns(m):
    return m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec


class Recorder(Node):
    def __init__(self):
        super().__init__('stamp_trace_recorder')
        self.rows = []
        self.create_subscription(Imu, IMU_TOPIC, self.on_imu,
                                 QoSProfile(depth=500, reliability=ReliabilityPolicy.RELIABLE))

    def on_imu(self, m):
        a, w = m.linear_acceleration, m.angular_velocity
        self.rows.append(f'I,{stamp_ns(m)},{a.x!r},{a.y!r},{a.z!r},{w.x!r},{w.y!r},{w.z!r}')


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    out = open(sys.argv[2], 'w') if len(sys.argv) > 2 else sys.stdout
    rclpy.init()
    node = Recorder()
    end = time.time() + duration
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.01)
    out.write('\n'.join(node.rows) + '\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
