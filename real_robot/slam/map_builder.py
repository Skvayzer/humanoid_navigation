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
While mapping it also autosaves these files every 30 s (a dead battery no longer loses the map) and publishes the
map so far on /g1_slam/map (10 cm voxels, camera_init frame, every 2 s) for Foxglove.
"""
import array
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from slam_common import cloud_xyz, fit_floor, level_transform, matrix_to_quat, pose_to_matrix, write_pcd

VOXEL = 0.05
VIEW_VOXEL = 0.10            # /g1_slam/map: coarser, so it fits through rosbridge over WiFi
PUBLISH_S, AUTOSAVE_S = 2.0, 30.0


class Accumulator:
    """5 cm voxel map: per voxel the sum of point positions and the number of scans that hit it. Scans are queued
    and merged in batches (compact), so memory and merge time grow with the room, not with the walk."""

    def __init__(self, voxel=VOXEL):
        self.voxel = voxel; self.pending = []; self.count = 0; self.scans = 0
        self.keys = np.zeros((0, 3), np.int64); self.sums = np.zeros((0, 3)); self.hits = np.zeros(0, np.int64)

    def add(self, pts):
        k = np.floor(pts / self.voxel).astype(np.int64)
        _, first = np.unique(k, axis=0, return_index=True)          # one point per voxel and scan
        self.pending.append((k[first], pts[first])); self.count += len(first); self.scans += 1

    def compact(self):
        if not self.pending:
            return
        k = np.concatenate([self.keys] + [p[0] for p in self.pending])
        s = np.concatenate([self.sums] + [p[1] for p in self.pending])
        h = np.concatenate([self.hits] + [np.ones(len(p[0]), np.int64) for p in self.pending])
        self.pending = []
        self.keys, inverse = np.unique(k, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        self.sums = np.zeros((len(self.keys), 3)); np.add.at(self.sums, inverse, s)
        self.hits = np.bincount(inverse, weights=h, minlength=len(self.keys)).astype(np.int64)

    def merged(self):
        self.compact()
        return (self.sums / np.maximum(self.hits, 1)[:, None]).astype(np.float32), self.hits.astype(np.int32)


def view(pts, hits, voxel=VIEW_VOXEL, min_hits=2):
    """One point per coarse voxel (the first), only voxels seen in >= min_hits scans."""
    pts = pts[hits >= min_hits]
    _, first = np.unique(np.floor(pts / voxel).astype(np.int64), axis=0, return_index=True)
    return pts[first]


def _replace_into(out_dir, files):
    for tmp, final in files:                                         # each file is whole even if power dies now
        os.replace(tmp, final)


def save(out_dir, name, pts_ci, hits, path, t_map_ci, min_hits=2):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    keep = hits >= min_hits                                           # drop single-scan returns (people, noise)
    pts_ci, hits = pts_ci[keep], hits[keep]
    done = []
    tmp = lambda suffix: out_dir / ('.%s%s.tmp' % (name, suffix))
    write_pcd(tmp('_raw.pcd'), pts_ci, hits); done.append((tmp('_raw.pcd'), out_dir / (name + '_raw.pcd')))
    pts = pts_ci @ t_map_ci[:3, :3].T + t_map_ci[:3, 3]
    a, b, c, n, rms = fit_floor(pts, z_guess=0.0, band=0.3, r_max=15.0)
    level = level_transform(a, b, c)                                   # refine on the whole map
    t_map_ci = level @ t_map_ci
    pts = pts_ci @ t_map_ci[:3, :3].T + t_map_ci[:3, 3]
    write_pcd(tmp('.pcd'), pts, hits); done.append((tmp('.pcd'), out_dir / (name + '.pcd')))
    start = t_map_ci @ path[0][1]
    q = matrix_to_quat(start[:3, :3])
    rows = []
    for t, pose in path:
        m = t_map_ci @ pose; qq = matrix_to_quat(m[:3, :3]); rows.append([t, *m[:3, 3], *qq])
    np.savetxt(tmp('_path.csv'), np.array(rows), delimiter=',', header='t,x,y,z,qx,qy,qz,qw', comments='')
    done.append((tmp('_path.csv'), out_dir / (name + '_path.csv')))
    meta = {'/**': {'ros__parameters': {
        'map_path': '/data/maps/%s.pcd' % name, 'set_initial_pose': True,
        'initial_pose_x': float(start[0, 3]), 'initial_pose_y': float(start[1, 3]), 'initial_pose_z': float(start[2, 3]),
        'initial_pose_qx': float(q[0]), 'initial_pose_qy': float(q[1]), 'initial_pose_qz': float(q[2]), 'initial_pose_qw': float(q[3])}}}
    import yaml
    tmp('.yaml').write_text(yaml.safe_dump(meta, sort_keys=False)); done.append((tmp('.yaml'), out_dir / (name + '.yaml')))
    info = dict(points=int(len(pts)), floor_refine=dict(a=a, b=b, c=c, points=n, rms_m=rms), scans=len(path),
                path_length_m=float(np.sum(np.linalg.norm(np.diff(np.array([r[1:4] for r in rows]), axis=0), axis=1))) if len(rows) > 1 else 0.,
                t_map_camera_init=t_map_ci.tolist())
    tmp('_info.json').write_text(json.dumps(info, indent=1)); done.append((tmp('_info.json'), out_dir / (name + '_info.json')))
    _replace_into(out_dir, done)
    return info


def lookup_map_ci(tf):
    from rclpy.time import Time
    from slam_common import quat_to_matrix
    tr = tf.lookup_transform('map', 'camera_init', Time()).transform
    t = np.eye(4)
    t[:3, :3] = quat_to_matrix(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
    t[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    return t


def cloud_msg(PointCloud2, PointField, stamp, frame, pts):
    m = PointCloud2(); m.header.frame_id = frame; m.header.stamp = stamp
    pts = np.ascontiguousarray(pts, dtype='<f4')
    m.height, m.width = 1, len(pts)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
    m.point_step, m.row_step, m.is_dense = 12, 12 * len(pts), True
    m.data = array.array('B', pts.tobytes())                          # Foxy: assigning bytes validates every element
    return m


def main():
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField
    from tf2_ros import Buffer, TransformListener
    out_dir, name = sys.argv[1], sys.argv[2]
    rclpy.init(); node = Node('map_builder', namespace='g1_slam', cli_args=['--ros-args', '-r', 'tf:=/tf', '-r', 'tf_static:=/tf_static']); log = node.get_logger()
    acc = Accumulator(); path = []; lock = threading.Lock(); stop = threading.Event()
    tf = Buffer(); TransformListener(tf, node)
    node.create_subscription(PointCloud2, '/g1_slam/cloud_registered', lambda m: (lock.acquire(), acc.add(cloud_xyz(m)), lock.release()), 5)
    node.create_subscription(Odometry, '/g1_slam/odometry', lambda m: path.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec,
                                                                                   pose_to_matrix(m.pose.pose.position, m.pose.pose.orientation))), 50)
    signal.signal(signal.SIGINT, lambda *_: stop.set()); signal.signal(signal.SIGTERM, lambda *_: stop.set())
    map_pub = node.create_publisher(PointCloud2, '/g1_slam/map', 1)
    log.info('mapping into %s/%s: walk the robot slowly, return to the start mark, then stop' % (out_dir, name))
    last_log = last_pub = last_save = time.time(); saving = None

    def autosave(pts, hits, path_copy):
        try:
            t0 = time.time(); info = save(out_dir, name, pts, hits, path_copy, lookup_map_ci(tf))
            log.info('autosaved %s: %d points, %.1f m path (%.1f s)' % (name, info['points'], info['path_length_m'], time.time() - t0))
        except Exception as exc:
            log.warn('autosave skipped: %s' % exc)

    while rclpy.ok() and not stop.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.time()
        if now - last_pub > PUBLISH_S:
            last_pub = now
            with lock:
                pts, hits = acc.merged()
            map_pub.publish(cloud_msg(PointCloud2, PointField, node.get_clock().now().to_msg(), 'camera_init', view(pts, hits)))
            if now - last_save > AUTOSAVE_S and path and (saving is None or not saving.is_alive()):
                last_save = now
                saving = threading.Thread(target=autosave, args=(pts, hits, list(path)), daemon=True); saving.start()
        if now - last_log > 10:
            last_log = now; log.info('%d scans, %d voxel samples, %d voxels' % (acc.scans, acc.count, len(acc.hits)))
    if saving is not None:
        saving.join()
    t_map_ci = None
    try:
        t_map_ci = lookup_map_ci(tf)
    except Exception as exc:
        log.error('no map -> camera_init (floor anchor): %s; the map is saved in camera_init only' % exc)
    with lock:
        pts, hits = acc.merged()
    if t_map_ci is None or not path:
        Path(out_dir).mkdir(parents=True, exist_ok=True); write_pcd(Path(out_dir) / (name + '_raw.pcd'), pts, hits)
    else:
        info = save(out_dir, name, pts, hits, path, t_map_ci)
        log.info('saved %s: %d points, %.1f m path, floor rms %.1f cm' % (name, info['points'], info['path_length_m'], 100 * info['floor_refine']['rms_m']))
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
