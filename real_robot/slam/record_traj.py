#!/usr/bin/env python3
"""Record the SLAM trajectory to CSV: stamp, odom pose (camera_init -> body) and, when TF has it, map -> body.

  record_traj.py out.csv          (stop with SIGINT/SIGTERM; writes on exit)
"""
import signal
import sys
import threading

import numpy as np

from slam_common import matrix_to_quat, pose_to_matrix, quat_to_matrix


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from nav_msgs.msg import Odometry
    from tf2_ros import Buffer, TransformListener
    rclpy.init(); node = Node('traj_recorder'); tf = Buffer(); TransformListener(tf, node)
    rows = []; stop = threading.Event()

    def on_odom(m):
        t_ci = pose_to_matrix(m.pose.pose.position, m.pose.pose.orientation)
        row = [m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, *t_ci[:3, 3], *matrix_to_quat(t_ci[:3, :3])]
        try:
            tr = tf.lookup_transform('map', 'camera_init', Time()).transform
            t_mc = np.eye(4); t_mc[:3, :3] = quat_to_matrix(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
            t_mc[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
            t_m = t_mc @ t_ci; row += [*t_m[:3, 3], *matrix_to_quat(t_m[:3, :3])]
        except Exception:
            row += [np.nan] * 7
        rows.append(row)

    node.create_subscription(Odometry, '/g1_slam/odometry', on_odom, 50)
    signal.signal(signal.SIGINT, lambda *_: stop.set()); signal.signal(signal.SIGTERM, lambda *_: stop.set())
    while rclpy.ok() and not stop.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)
    np.savetxt(sys.argv[1], np.array(rows) if rows else np.zeros((0, 15)), delimiter=',', comments='',
               header='t,x,y,z,qx,qy,qz,qw,map_x,map_y,map_z,map_qx,map_qy,map_qz,map_qw')
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
