#!/usr/bin/env python3
"""End-position error of a SLAM run on a bag that starts and ends standing still (as in the FAST-LIO report):
the true start->end motion comes from point-to-plane ICP between the first and last LiDAR scans.

  end_error.py <bag_dir> <trajectory.csv: t,x,y,z,qx,qy,qz,qw (body in any fixed frame)>   (needs rosbags, scipy)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from slam_common import T_EXT, quat_to_matrix


def load_scans(bag, n=10):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    scans = []
    with AnyReader([Path(bag) / 'bag'], default_typestore=get_typestore(Stores.ROS2_FOXY)) as r:
        conns = [c for c in r.connections if c.topic == '/utlidar/cloud_livox_mid360']
        for c, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            dt = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3, 'offsets': [0, 4, 8], 'itemsize': m.point_step})
            a = np.frombuffer(bytes(m.data), dt)
            p = np.column_stack([a['x'], a['y'], a['z']]).astype(float) + T_EXT                # body (IMU) frame
            d = np.linalg.norm(p, axis=1); scans.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, p[(d > 0.5) & (d < 20)]))
    return scans[:n], scans[-n:]


def icp(src, dst, t0, iters=40):
    tree = cKDTree(dst)
    _, nn = tree.query(dst, k=12)
    normals = np.empty_like(dst)
    for i, idx in enumerate(nn):                            # PCA normal per destination point
        q = dst[idx] - dst[idx].mean(0); normals[i] = np.linalg.svd(q, full_matrices=False)[2][2]
    t = t0.copy()
    for k in range(iters):
        p = src @ t[:3, :3].T + t[:3, 3]
        d, j = tree.query(p, distance_upper_bound=0.5 if k < 10 else 0.2)
        ok = np.isfinite(d); ps, qd, nd = p[ok], dst[j[ok]], normals[j[ok]]
        A = np.column_stack([np.cross(ps, nd), nd]); b = ((qd - ps) * nd).sum(1)
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        w = x[:3]; th = np.linalg.norm(w)
        r = np.eye(3) if th < 1e-12 else quat_to_matrix(*(w / th * np.sin(th / 2)), np.cos(th / 2))
        step = np.eye(4); step[:3, :3] = r; step[:3, 3] = x[3:]
        t = step @ t
        if np.linalg.norm(x) < 1e-6:
            break
    return t, float(np.sqrt(np.mean(((qd - ps) * nd).sum(1) ** 2))), int(ok.sum())


def pose(row):
    t = np.eye(4); t[:3, :3] = quat_to_matrix(*row[4:8]); t[:3, 3] = row[1:4]; return t


def main():
    bag, traj = sys.argv[1], np.loadtxt(sys.argv[2], delimiter=',', skiprows=1)
    traj = traj[np.isfinite(traj[:, 1])]
    first, last = load_scans(bag)
    start, end = pose(traj[0]), pose(traj[-1])
    rel_slam = np.linalg.inv(start) @ end                                 # end body pose in the start body frame
    dst = np.concatenate([p for _, p in first])[::2]; src = np.concatenate([p for _, p in last])[::2]
    rel_true, rms, used = icp(src, dst, rel_slam)
    err = rel_slam[:3, 3] - rel_true[:3, 3]
    ang = np.degrees(np.arccos(np.clip((np.trace(rel_true[:3, :3].T @ rel_slam[:3, :3]) - 1) / 2, -1, 1)))
    print('trajectory: %d poses over %.1f s; start->end displacement %.3f m' % (len(traj), traj[-1, 0] - traj[0, 0], np.linalg.norm(rel_true[:3, 3])))
    print('ICP first/last scans: rms %.3f m on %d points' % (rms, used))
    print('END ERROR: %.3f m (horizontal %.3f m), rotation %.2f deg' % (np.linalg.norm(err), np.linalg.norm(err[:2]), ang))


if __name__ == '__main__':
    main()
