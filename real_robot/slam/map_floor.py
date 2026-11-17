#!/usr/bin/env python3
"""The floor frame of a saved map, shared by localization and the static ESDF layer.

    map_floor.py /data/maps <name>

Inputs (one mapping session, run_slam.sh mapping <name>):
  <name>.vmap        FAST_LIO_G1's voxel map, what localization matches against. Its frame (slam_map) is the session's
                     slam_init rotated level by FAST-LIO's final gravity estimate (a tilt about the origin, heading kept).
  <name>_raw.pcd     map_builder's dense 5 cm cloud in camera_init (= slam_init unless the session had a jump)
  <name>_path.csv    + <name>_info.json: the trajectory and that session's map -> camera_init

Both clouds are put into "their floor frame" the same way (slam_common.level_transform: the smallest rotation that
makes the floor flat, about the origin, then z = 0 on the floor). Since the .vmap differs from the dense cloud's
frame only by a tilt about the same origin, the two floor frames agree to second order (well under a centimetre
over a room), and this script checks it. Outputs:
  <name>_floor.json  T_map_slam_map: published as map -> slam_map in localize sessions (floor_anchor.py --from-json)
  <name>_static.pcd  the dense cloud in that map frame, and <name>_static_path.csv the trajectory (x, y) in it,
                     for the static ESDF (wholebody esdf_layers)
"""
import json
import sys
from pathlib import Path

import numpy as np

from slam_common import fit_floor, inverse, level_transform, read_pcd, write_pcd

HEAD = np.dtype([('magic', 'S4'), ('version', '<u4'), ('mark', '<u4'), ('t', '<f8', 3), ('q', '<f8', 4),
                 ('params', '<f8', 7), ('count', '<u8')])
VOXEL = np.dtype([('key', '<u8'), ('n', '<i4'), ('w', '<f8'), ('w2', '<f8'), ('sum', '<f8', 3), ('sum_sq', '<f8', 9)])


def read_vmap_means(path):
    """Voxel means of a .vmap (version 3), in the file's own frame, and the voxel size."""
    raw = Path(path).read_bytes()
    h = np.frombuffer(raw, HEAD, 1)[0]
    if h['magic'] != b'VXMP' or h['version'] != 3 or h['mark'] != 0x01020304:
        raise ValueError('%s: not a version-3 little-endian .vmap' % path)
    v = np.frombuffer(raw, VOXEL, int(h['count']), HEAD.itemsize)
    k = v['key'].astype(np.int64); off = 1 << 20
    idx = np.stack([((k >> 42) & 0x1FFFFF) - off, ((k >> 21) & 0x1FFFFF) - off, (k & 0x1FFFFF) - off], 1)
    size = float(h['params'][0])
    return idx * size + v['sum'] / v['w'][:, None], size, idx


def floor_frame(points):
    a, b, c, n, rms = fit_floor(points, z_guess=-1.2, band=0.6, r_max=15.0)
    return level_transform(a, b, c), dict(points=n, rms_m=rms, slope_deg=float(np.degrees(np.arctan(np.hypot(a, b)))),
                                          height_m=-c)


def apply(t, p):
    return p @ t[:3, :3].T + t[:3, 3]


def main():
    maps, name = Path(sys.argv[1]), sys.argv[2]
    means, size, keys = read_vmap_means(maps / (name + '.vmap'))
    t_vmap, fv = floor_frame(means)
    c = read_pcd(maps / (name + '_raw.pcd')); raw = np.column_stack([c['x'], c['y'], c['z']]).astype(float)
    t_raw, fr = floor_frame(raw)
    static = apply(t_raw, raw)
    # people / the operator: the dense cloud keeps whatever 2 scans saw; the .vmap's free space removed what moved.
    # Keep the dense points whose .vmap voxel survived (dense floor frame -> .vmap frame: the two floor frames agree)
    in_vmap = apply(inverse(t_vmap), static)
    vk = np.floor(in_vmap / size).astype(np.int64)
    pack = lambda k: ((k[:, 0] + (1 << 20)) << 42) | ((k[:, 1] + (1 << 20)) << 21) | (k[:, 2] + (1 << 20))
    kept = np.isin(pack(vk), pack(keys))
    removed = int((~kept).sum()); static = static[kept]
    # agreement: the .vmap's voxel means in its floor frame against the dense cloud in its own
    from scipy.spatial import cKDTree
    d, _ = cKDTree(static).query(apply(t_vmap, means), distance_upper_bound=1.0)
    d = d[np.isfinite(d)]
    path = np.loadtxt(maps / (name + '_path.csv'), delimiter=',', skiprows=1, ndmin=2)[:, 1:4]
    t_map_ci = np.array(json.loads((maps / (name + '_info.json')).read_text())['t_map_camera_init'])
    path = apply(t_raw @ inverse(t_map_ci), path)                      # mapping session's map -> camera_init -> floor
    write_pcd(maps / (name + '_static.pcd'), static)
    np.savetxt(maps / (name + '_static_path.csv'), np.column_stack([np.arange(len(path)), path[:, :2]]),
               delimiter=',', header='t,x,y', comments='')
    info = dict(T_map_slam_map=t_vmap.tolist(), vmap_floor=fv, dense_floor=fr, voxel_size_m=size,
                vmap_vs_dense_median_m=float(np.median(d)), vmap_vs_dense_p90_m=float(np.percentile(d, 90)),
                dense_points_removed_by_free_space=removed)
    (maps / (name + '_floor.json')).write_text(json.dumps(info, indent=1))
    print('%s: floor %.3f m below the .vmap origin (rms %.1f cm), dense cloud %d points; .vmap voxel means vs dense '
          'cloud: median %.1f cm, p90 %.1f cm; %d dense points not in the .vmap (moved: people) dropped'
          % (name, fv['height_m'], 100 * fv['rms_m'], len(raw), 100 * info['vmap_vs_dense_median_m'],
             100 * info['vmap_vs_dense_p90_m'], removed))


if __name__ == '__main__':
    main()
