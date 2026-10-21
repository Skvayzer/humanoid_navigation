#!/usr/bin/env python3
"""Dense map of the room from FAST-LIO, saved floor-aligned for NDT localisation.

  map_builder.py <out_dir> <name>      (mapping mode of run_slam.sh; stop with SIGINT/SIGTERM to save)

Accumulates /g1_slam/cloud_registered (deskewed scans in camera_init; dense_publish_en) on a 5 cm voxel grid and
the FAST-LIO trajectory. On stop it writes, in out_dir:
  <name>.pcd        map in the floor-aligned `map` frame (floor anchor of this session, refined on the whole map)
  <name>_raw.pcd    the same points in camera_init
  <name>.yaml       NDT parameters: map path, initial pose = where the robot stood when mapping started
  <name>_path.csv   trajectory (t, x, y, z, qx, qy, qz, qw) in map
Start the next session at the SAME floor mark, facing the same way, so the initial pose holds.
"""
import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from slam_common import cloud_xyz, fit_floor, level_transform, matrix_to_quat, pose_to_matrix, write_pcd

VOXEL = 0.05


class Accumulator:
    def __init__(self, voxel=VOXEL):
        self.voxel = voxel; self.keys = []; self.points = []; self.count = 0

    def add(self, pts):
        k = np.floor(pts / self.voxel).astype(np.int64)
        _, first = np.unique(k, axis=0, return_index=True)          # one point per voxel and scan
        self.keys.append(k[first]); self.points.append(pts[first].astype(np.float32)); self.count += len(first)

    def merged(self):
        if not self.points:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.int32)
        k = np.concatenate(self.keys); p = np.concatenate(self.points)
        uniq, inverse, hits = np.unique(k, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((len(uniq), 3)); np.add.at(sums, inverse, p)
        return (sums / hits[:, None]).astype(np.float32), hits.astype(np.int32)


def save(out_dir, name, acc, path, t_map_ci, min_hits=2):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pts_ci, hits = acc.merged()
    keep = hits >= min_hits                                           # drop single-scan returns (people, noise)
    pts_ci, hits = pts_ci[keep], hits[keep]
    write_pcd(out_dir / (name + '_raw.pcd'), pts_ci, hits)
    pts = pts_ci @ t_map_ci[:3, :3].T + t_map_ci[:3, 3]
    a, b, c, n, rms = fit_floor(pts, z_guess=0.0, band=0.3, r_max=15.0)
    level = level_transform(a, b, c)                                   # refine on the whole map
    t_map_ci = level @ t_map_ci
    pts = pts_ci @ t_map_ci[:3, :3].T + t_map_ci[:3, 3]
    write_pcd(out_dir / (name + '.pcd'), pts, hits)
    start = t_map_ci @ path[0][1]
    q = matrix_to_quat(start[:3, :3])
    rows = []
    for t, pose in path:
        m = t_map_ci @ pose; qq = matrix_to_quat(m[:3, :3]); rows.append([t, *m[:3, 3], *qq])
    np.savetxt(out_dir / (name + '_path.csv'), np.array(rows), delimiter=',', header='t,x,y,z,qx,qy,qz,qw', comments='')
    meta = {'/**': {'ros__parameters': {
        'map_path': '/data/maps/%s.pcd' % name, 'set_initial_pose': True,
        'initial_pose_x': float(start[0, 3]), 'initial_pose_y': float(start[1, 3]), 'initial_pose_z': float(start[2, 3]),
        'initial_pose_qx': float(q[0]), 'initial_pose_qy': float(q[1]), 'initial_pose_qz': float(q[2]), 'initial_pose_qw': float(q[3])}}}
    import yaml
    (out_dir / (name + '.yaml')).write_text(yaml.safe_dump(meta, sort_keys=False))
    info = dict(points=int(len(pts)), floor_refine=dict(a=a, b=b, c=c, points=n, rms_m=rms), scans=len(path),
                path_length_m=float(np.sum(np.linalg.norm(np.diff(np.array([r[1:4] for r in rows]), axis=0), axis=1))) if len(rows) > 1 else 0.,
                t_map_camera_init=t_map_ci.tolist())
    (out_dir / (name + '_info.json')).write_text(json.dumps(info, indent=1))
    return info


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from tf2_ros import Buffer, TransformListener
    out_dir, name = sys.argv[1], sys.argv[2]
    rclpy.init(); node = Node('map_builder', namespace='g1_slam', cli_args=['--ros-args', '-r', 'tf:=/tf', '-r', 'tf_static:=/tf_static']); log = node.get_logger()
    acc = Accumulator(); path = []; lock = threading.Lock(); stop = threading.Event()
    tf = Buffer(); TransformListener(tf, node)
    node.create_subscription(PointCloud2, '/g1_slam/cloud_registered', lambda m: (lock.acquire(), acc.add(cloud_xyz(m)), lock.release()), 5)
    node.create_subscription(Odometry, '/g1_slam/odometry', lambda m: path.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec,
                                                                                   pose_to_matrix(m.pose.pose.position, m.pose.pose.orientation))), 50)
    signal.signal(signal.SIGINT, lambda *_: stop.set()); signal.signal(signal.SIGTERM, lambda *_: stop.set())
    log.info('mapping into %s/%s: walk the robot slowly, return to the start mark, then stop' % (out_dir, name))
    last = time.time()
    while rclpy.ok() and not stop.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - last > 10:
            last = time.time(); log.info('%d scans, %d voxel samples' % (len(acc.points), acc.count))
    t_map_ci = None
    try:
        tr = tf.lookup_transform('map', 'camera_init', Time()).transform
        t_map_ci = np.eye(4)
        from slam_common import quat_to_matrix
        t_map_ci[:3, :3] = quat_to_matrix(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
        t_map_ci[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    except Exception as exc:
        log.error('no map -> camera_init (floor anchor): %s; the map is saved in camera_init only' % exc)
    with lock:
        if t_map_ci is None or not path:
            from slam_common import write_pcd as wp
            p, h = acc.merged(); Path(out_dir).mkdir(parents=True, exist_ok=True); wp(Path(out_dir) / (name + '_raw.pcd'), p, h)
        else:
            info = save(out_dir, name, acc, path, t_map_ci)
            log.info('saved %s: %d points, %.1f m path, floor rms %.1f cm' % (name, info['points'], info['path_length_m'], 100 * info['floor_refine']['rms_m']))
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
