#!/usr/bin/env python3
"""Light clouds from FAST-LIO's scans, for the viewer and the executor (the full scans are too heavy for rosbridge):

  /g1_slam/map        the room so far: 10 cm voxels seen in >= 2 scans, camera_init, every 2 s
                      (not with --no-map: in mapping mode map_builder publishes it)
  /g1_slam/obstacles  what is around the robot now: scans of the last 1.5 s, 4 cm voxels within 4 m of the
                      sensor, camera_init, 4 Hz (the executor's 'real' obstacle source)

  cloud_relay.py [--no-map]          (started by run_slam.sh in every mode)
"""
import collections
import sys
import time

import numpy as np

from map_builder import Accumulator, cloud_msg, view
from slam_common import cloud_xyz

MAP_VOXEL, MAP_PERIOD_S = 0.10, 2.0
OBST_VOXEL, OBST_WINDOW_S, OBST_RADIUS_M, OBST_PERIOD_S = 0.04, 1.5, 4.0, 0.25


def voxel_down(pts, voxel):
    if len(pts) == 0:                                   # old numpy: unique(axis=0) fails on empty input
        return pts
    _, first = np.unique(np.floor(pts / voxel).astype(np.int64), axis=0, return_index=True)
    return pts[first]


def main():
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField
    publish_map = '--no-map' not in sys.argv
    rclpy.init(); node = Node('cloud_relay', namespace='g1_slam'); log = node.get_logger()
    acc = Accumulator(voxel=MAP_VOXEL); recent = collections.deque(); sensor = [None]

    def on_scan(msg):
        pts = cloud_xyz(msg)
        if len(pts) == 0:
            return
        if publish_map:
            acc.add(pts)
        recent.append((time.monotonic(), voxel_down(pts, OBST_VOXEL)))

    def on_odom(msg):
        p = msg.pose.pose.position; sensor[0] = np.array([p.x, p.y, p.z])

    node.create_subscription(PointCloud2, '/g1_slam/cloud_registered', on_scan, 5)
    node.create_subscription(Odometry, '/g1_slam/odometry', on_odom, 10)
    map_pub = node.create_publisher(PointCloud2, '/g1_slam/map', 1)
    obst_pub = node.create_publisher(PointCloud2, '/g1_slam/obstacles', 1)
    log.info('relaying /g1_slam/obstacles%s' % (' and /g1_slam/map' if publish_map else ''))
    last_map = last_obst = 0.
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.monotonic()
        while recent and now - recent[0][0] > OBST_WINDOW_S:
            recent.popleft()
        if now - last_obst > OBST_PERIOD_S and recent and sensor[0] is not None:
            last_obst = now
            pts = voxel_down(np.concatenate([p for _, p in recent]), OBST_VOXEL)
            pts = pts[np.linalg.norm(pts[:, :2] - sensor[0][:2], axis=1) < OBST_RADIUS_M]
            obst_pub.publish(cloud_msg(PointCloud2, PointField, node.get_clock().now().to_msg(), 'camera_init', pts))
        if publish_map and now - last_map > MAP_PERIOD_S:
            last_map = now
            pts, hits = acc.merged()
            map_pub.publish(cloud_msg(PointCloud2, PointField, node.get_clock().now().to_msg(), 'camera_init',
                                      view(pts, hits, voxel=MAP_VOXEL)))
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
