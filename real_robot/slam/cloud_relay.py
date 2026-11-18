#!/usr/bin/env python3
"""Light clouds from FAST-LIO's scans, for the viewer and the executor (the full scans are too heavy for rosbridge):

  /g1_slam/map        the room so far: 10 cm voxels seen in >= 2 scans, camera_init, every 2 s
                      (not with --no-map: in mapping mode map_builder publishes it)
  /g1_slam/obstacles  what is around the robot now: scans of the last 0.5 s, 4 cm voxels within 4 m of the
                      sensor, camera_init, 10 Hz (the executor's 'real' obstacle source); fields x y z t, t = the time
                      of the point's scan on the system clock minus the header stamp (s, <= 0), for the self-filter

  cloud_relay.py [--no-map]          (started by run_slam.sh in every mode)
"""
import array
import collections
import signal
import sys
import time

import numpy as np

from map_builder import Accumulator, cloud_msg, view
from slam_common import cloud_xyz

MAP_VOXEL, MAP_PERIOD_S = 0.10, 2.0
OBST_VOXEL, OBST_WINDOW_S, OBST_RADIUS_M, OBST_PERIOD_S = 0.04, 0.5, 4.0, 0.1
SCAN_HALF_S = 0.05                                     # half a 10 Hz scan


def voxel_down(pts, voxel):
    """One point per voxel: the first one (put the newest first to keep the newest). Extra columns ride along."""
    if len(pts) == 0:                                   # old numpy: unique(axis=0) fails on empty input
        return pts
    _, first = np.unique(np.floor(pts[:, :3] / voxel).astype(np.int64), axis=0, return_index=True)
    return pts[first]


def xyzt_msg(PointCloud2, PointField, stamp, frame, pts):
    m = PointCloud2(); m.header.frame_id = frame; m.header.stamp = stamp
    pts = np.ascontiguousarray(pts, dtype='<f4')
    m.height, m.width = 1, len(pts)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyzt')]
    m.point_step, m.row_step, m.is_dense = 16, 16 * len(pts), True
    m.data = array.array('B', pts.tobytes())
    return m


def main():
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField
    publish_map = '--no-map' not in sys.argv
    rclpy.init(); node = Node('cloud_relay', namespace='g1_slam'); log = node.get_logger()
    acc = Accumulator(voxel=MAP_VOXEL); recent = collections.deque(); sensor = [None]
    # FAST-LIO stamps follow the lidar's clock (~20 min off the system clock on the G1): the offset is the smallest
    # (system receive - stamp) of the recent odometry, i.e. the clock offset plus the minimal latency (~1 ms)
    offsets = collections.deque(maxlen=200)
    stamp_s = lambda st: st.sec + 1e-9 * st.nanosec

    def on_scan(msg):
        pts = cloud_xyz(msg)
        if len(pts) == 0:
            return
        if publish_map:
            acc.add(pts)
        # FAST-LIO stamps a scan with its end; its points span the 0.1 s before: the centre, +-0.05 s (the executor's
        # self-filter tests the body over that interval)
        t_scan = stamp_s(msg.header.stamp) + (min(offsets) if offsets else 0.) - SCAN_HALF_S
        if not offsets:                                  # no odometry yet: the receive time
            t_scan = time.time()
        recent.append((time.monotonic(), voxel_down(pts, OBST_VOXEL), t_scan))

    def on_odom(msg):
        p = msg.pose.pose.position; sensor[0] = np.array([p.x, p.y, p.z])
        offsets.append(time.time() - stamp_s(msg.header.stamp))

    node.create_subscription(PointCloud2, '/g1_slam/cloud_registered', on_scan, 5)
    node.create_subscription(Odometry, '/g1_slam/odometry', on_odom, 10)
    map_pub = node.create_publisher(PointCloud2, '/g1_slam/map', 1)
    obst_pub = node.create_publisher(PointCloud2, '/g1_slam/obstacles', 1)
    log.info('relaying /g1_slam/obstacles%s' % (' and /g1_slam/map' if publish_map else ''))
    last_map = last_obst = 0.
    # started with '&' by run_slam.sh (non-interactive bash): SIGINT arrives ignored, so set the handlers ourselves,
    # or 'g1.sh slam stop' waits the full docker stop timeout
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1)); signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    while rclpy.ok() and not stop:
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.monotonic()
        while recent and now - recent[0][0] > OBST_WINDOW_S:
            recent.popleft()
        if now - last_obst > OBST_PERIOD_S and recent and sensor[0] is not None:
            last_obst = now
            stamp = node.get_clock().now(); t_now = stamp.nanoseconds * 1e-9
            pts = np.concatenate([np.column_stack([p, np.full(len(p), t - t_now)]) for _, p, t in reversed(recent)])
            pts = voxel_down(pts, OBST_VOXEL)                # newest scans first: each voxel keeps its newest time
            pts = pts[np.linalg.norm(pts[:, :2] - sensor[0][:2], axis=1) < OBST_RADIUS_M]
            obst_pub.publish(xyzt_msg(PointCloud2, PointField, stamp.to_msg(), 'camera_init', pts))
        if publish_map and now - last_map > MAP_PERIOD_S and map_pub.get_subscription_count():   # merge only for a viewer
            last_map = now
            pts, hits = acc.merged()
            map_pub.publish(cloud_msg(PointCloud2, PointField, node.get_clock().now().to_msg(), 'camera_init',
                                      view(pts, hits, voxel=MAP_VOXEL)))
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
