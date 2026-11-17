"""Converts a .vmap (FAST_LIO's voxel map, map_save) into a PCD for point cloud viewers (CloudCompare, pcl_viewer).

    python scripts/vmap_to_pcd.py <map.vmap> [<out.pcd>] [--own-frame]

One point per voxel at its mean, with the fields points (observations), normal_x/y/z and planar (the voxel's
plane: 1 if its distribution is one surface, VoxelMap::one_surface). Positions are in the primary map's frame: a
live layer saved in localization mode (the file holds map -> init) lands on its primary map; --own-frame keeps the
file's own frame. Output: <map>.pcd unless given.
The .vmap format (version 3, little-endian here) is described in include/voxel_map.h.
"""
import argparse
import struct
import sys
import numpy as np


def read_vmap(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'VXMP':
        sys.exit('%s: not a .vmap' % path)
    version, mark = struct.unpack_from('<II', data, 4)
    if version != 3:
        sys.exit('%s: a .vmap of version %d, this script reads version 3' % (path, version))
    if mark != 0x01020304:
        sys.exit('%s: written on a machine with another byte order' % path)
    off = 12
    t = np.array(struct.unpack_from('<3d', data, off)); off += 24
    qx, qy, qz, qw = struct.unpack_from('<4d', data, off); off += 32
    size, point_sigma, shape_ratio, min_n_eff, freeze_sigma, pooled_thickness, pooled_planarity = \
        struct.unpack_from('<7d', data, off)
    off += struct.calcsize('<7d')
    n_voxels, = struct.unpack_from('<Q', data, off); off += 8
    head = struct.Struct('<Qidd3d9d')   # key, n, w, w2, sum, sum_sq
    voxels = []
    for _ in range(n_voxels):
        rec = head.unpack_from(data, off); off += head.size
        key, n, w, w2 = rec[0], rec[1], rec[2], rec[3]
        s = np.array(rec[4:7]); ss = np.array(rec[7:16]).reshape(3, 3)
        # the plane of the voxel's distribution (VoxelMap::one_surface): S + sigma^2 / n_eff I
        m = s / w
        cov = ss / w - np.outer(m, m) + (point_sigma ** 2 * w2 / (w * w)) * np.eye(3)
        l, e = np.linalg.eigh(cov)
        normal = e[:, 0]
        valid = l[0] <= pooled_thickness ** 2 and l[0] <= pooled_planarity * l[1]
        idx = np.array([((key >> 42) & 0x1FFFFF) - (1 << 20), ((key >> 21) & 0x1FFFFF) - (1 << 20),
                        (key & 0x1FFFFF) - (1 << 20)], dtype=float)
        voxels.append((idx * size + m, n, normal, valid))   # mean as VoxelMap::to_cloud
    R = quat_to_matrix(qx, qy, qz, qw)
    return voxels, R, t, size


def quat_to_matrix(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def write_pcd(path, fields, columns):
    a = np.stack(columns, axis=1).astype('<f4')
    header = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS %s\nSIZE %s\nTYPE %s\nCOUNT %s\n'
              'WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n' % (
                  ' '.join(fields), ' '.join(['4'] * len(fields)), ' '.join(['F'] * len(fields)),
                  ' '.join(['1'] * len(fields)), len(a), len(a)))
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('vmap')
    ap.add_argument('pcd', nargs='?')
    ap.add_argument('--own-frame', action='store_true', help="keep the file's own frame")
    args = ap.parse_args()
    out = args.pcd or (args.vmap[:-5] if args.vmap.endswith('.vmap') else args.vmap) + '.pcd'
    voxels, R, t, size = read_vmap(args.vmap)
    if args.own_frame:
        R, t = np.eye(3), np.zeros(3)
    M = np.array([v[0] for v in voxels]) @ R.T + t
    N = np.array([v[2] for v in voxels]) @ R.T
    write_pcd(out, ['x', 'y', 'z', 'points', 'normal_x', 'normal_y', 'normal_z', 'planar'],
              [M[:, 0], M[:, 1], M[:, 2], np.array([v[1] for v in voxels]), N[:, 0], N[:, 1], N[:, 2],
               np.array([v[3] for v in voxels])])
    print('%s: %d voxel means (%.2f m)' % (out, len(M), size))


if __name__ == '__main__':
    main()
