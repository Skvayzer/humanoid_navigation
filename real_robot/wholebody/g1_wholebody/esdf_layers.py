"""Real obstacles for the policy's fields in two layers, so that nothing slow runs while the robot walks.

  static   a prebuilt map of the room (map_builder's <name>.pcd): one signed distance field for the whole room, made
           offline (build_static, below) with the same skfmm sdf as training; at run time a window is just a crop.
  dynamic  the lidar points of the last ~0.5 s (cloud_relay /g1_slam/obstacles) minus the robot's own body and the
           gantry rope, minus what the static layer already explains: usually a few hundred voxels (a person, a
           moved chair). Their distance is stamped from a precomputed ball, truncated at DYN_RADIUS_M (the policy
           clips the sdf it observes at 0.5 m), ~10-20 ms instead of a 1-2 s FMM of the whole window.

  sdf = min(base, dynamic): base = static crop (and/or the virtual boxes); the window's guidance keeps the slow
  geodesic to the goal (fields.FieldWorker re-solves it ~1 Hz) and only its blending at walls is redone with the
  fresh sdf (wholebody_fields.mix_guidance), so obstacles reach the policy at the lidar rate.

  python -m g1_wholebody.esdf_layers <map.pcd> <out.npz> [--path <name>_path.csv] [--extent XMIN XMAX YMIN YMAX]
"""
import argparse
import math

import numpy as np

from wholebody_sim import DX, WINDOW

REAL_Z = (0.10, 1.40)          # obstacle band: above the floor, below the window top (as sim's robot_perception)
DYN_RADIUS_M = 0.52            # dynamic distance truncation (just above the policy's 0.5 m sdf clip)
STATIC_MATCH_M = 0.08          # a lidar voxel this close to the base layer's surface is the base layer (localisation error)
BODY_RADIUS_M = 0.12           # around each limb segment / field site
TRUNK_RADIUS_M = 0.20          # pelvis-torso segment
ROPE_RADIUS_M, ROPE_FROM_Z = 0.30, 1.00     # gantry rope: a column above the pelvis
STAMP_MAX_VOXELS = 3000        # more dynamic voxels than this are stamped from 8 cm centres (marked occupied at 4 cm)
FREE_SDF = float(np.linalg.norm(np.array(WINDOW) * DX))     # training's sdf of an empty window


def fmm_sdf(blocked, dx=DX):
    """The training sdf of an occupancy (wholebody_fields.training_fields)."""
    import skfmm
    if not blocked.any():
        return np.full(blocked.shape, FREE_SDF, np.float32)
    return skfmm.distance(np.where(blocked, -1., 1.), dx=dx).astype(np.float32)


# ------------------------------------------------------------------------------------------------ static layer
def read_pcd(path):
    """x, y, z of a binary or ascii PCD with float32 x y z first (map_builder / FAST-LIO)."""
    raw = open(path, 'rb').read(); end = raw.index(b'DATA ') ; line_end = raw.index(b'\n', end)
    header = raw[:end].decode('ascii', 'replace').split('\n'); kind = raw[end + 5:line_end].strip()
    h = {ln.split()[0]: ln.split()[1:] for ln in header if ln.strip() and not ln.startswith('#')}
    fields, sizes, n = h['FIELDS'], [int(s) for s in h['SIZE']], int(h['POINTS'][0])
    if fields[:3] != ['x', 'y', 'z'] or set(h['TYPE'][:3]) != {'F'} or sizes[:3] != [4, 4, 4]:
        raise ValueError('%s: expected float32 x y z first' % path)
    if kind == b'binary':
        return np.frombuffer(raw[line_end + 1:line_end + 1 + n * sum(sizes)], np.uint8).reshape(n, sum(sizes))[:, :12].copy().view('<f4')
    if kind == b'ascii':
        return np.loadtxt(raw[line_end + 1:].decode().splitlines(), dtype=np.float32, ndmin=2)[:, :3]
    raise ValueError('%s: unsupported PCD data %s' % (path, kind))


def read_path_xy(path):
    """Trajectory csv (map_builder <name>_path.csv: t,x,y,z,..., or record_traj with map_x/map_y) -> map xy."""
    with open(path) as f:
        head = f.readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    cols = (head.index('map_x'), head.index('map_y')) if 'map_x' in head else (1, 2)
    xy = data[:, cols]
    return xy[np.isfinite(xy).all(1)]


def build_static(points, extent, path_xy=None, clear_radius=0.35, z_range=REAL_Z, dx=DX):
    """Map points (map frame, floor at z = 0) -> dict(sdf [X, Y, Z] float32, first = centre of voxel 0, dx).
    The grid is on the windows' lattice (centres at (k + 1/2) dx, z from dx/2, WINDOW[2] layers), so a window is an
    exact crop. Points within clear_radius of the mapping path are dropped: the robot stood there, anything seen
    there is its own body or the gantry rope."""
    x0, x1, y0, y1 = extent
    first = np.array([(math.floor(x0 / dx) + .5) * dx, (math.floor(y0 / dx) + .5) * dx, dx / 2])
    shape = (int(math.ceil((x1 - first[0]) / dx)) + 1, int(math.ceil((y1 - first[1]) / dx)) + 1, WINDOW[2])
    p = np.asarray(points, float)
    p = p[(p[:, 2] > z_range[0]) & (p[:, 2] < z_range[1])]
    if path_xy is not None and len(path_xy):
        from scipy.spatial import cKDTree
        near, _ = cKDTree(np.asarray(path_xy)[:, :2]).query(p[:, :2], distance_upper_bound=clear_radius)
        p = p[~np.isfinite(near)]
    idx = np.floor((p - (first - dx / 2)) / dx).astype(int)
    idx = idx[(idx >= 0).all(1) & (idx < np.array(shape)).all(1)]
    occ = np.zeros(shape, bool); occ[tuple(idx.T)] = True
    return dict(sdf=fmm_sdf(occ, dx), first=first, dx=np.float64(dx), occupied=np.int64(occ.sum()))


class StaticEsdf:
    """A prebuilt static sdf (build_static's npz); crop() is a window of it, free (FREE_SDF) outside the map."""

    def __init__(self, path):
        d = np.load(path)
        self.sdf, self.first, self.dx = d['sdf'].astype(np.float32), d['first'].astype(float), float(d['dx'])
        if abs(self.dx - DX) > 1e-9 or self.sdf.shape[2] != WINDOW[2] or abs(self.first[2] - DX / 2) > 1e-6:
            raise ValueError('%s: not on the window lattice (dx %.3f, %d layers)' % (path, self.dx, self.sdf.shape[2]))
        self.path = str(path)

    def crop(self, first, shape):
        out = np.full(shape, FREE_SDF, np.float32)
        off = np.round((np.asarray(first, float)[:2] - self.first[:2]) / DX).astype(int)
        lo = np.maximum(off, 0); hi = np.minimum(off + np.array(shape[:2]), self.sdf.shape[:2])
        if (hi > lo).all():
            out[lo[0] - off[0]:hi[0] - off[0], lo[1] - off[1]:hi[1] - off[1]] = self.sdf[lo[0]:hi[0], lo[1]:hi[1]]
        return out


# ------------------------------------------------------------------------------------------------ the robot itself
def body_capsules(kin, q, pos, quat):
    """Segments (a, b) and radii covering the robot (limb segments between body origins, the field sites) and the
    gantry rope above it, in the map frame: lidar points inside them are not obstacles."""
    from wholebody_core import quat_matrix
    bodies, sites, _ = kin.frames(q)
    base = np.eye(4); base[:3, :3] = quat_matrix(quat); base[:3, 3] = pos
    at = lambda t: (base @ t)[:3, 3]
    a, b, r = [], [], []
    for name, info in kin.bodies.items():
        if info['parent'] is not None:
            a.append(at(bodies[info['parent']])); b.append(at(bodies[name]))
            trunk = any(k in name for k in ('pelvis', 'torso', 'waist'))
            r.append(TRUNK_RADIUS_M if trunk else BODY_RADIUS_M)
    for t in sites.values():
        p = at(t); a.append(p); b.append(p); r.append(BODY_RADIUS_M)
    p = np.asarray(pos, float)
    a.append(np.r_[p[:2], ROPE_FROM_Z]); b.append(np.r_[p[:2], 5.]); r.append(ROPE_RADIUS_M)
    return np.array(a), np.array(b), np.array(r)


def band(points, z_range=REAL_Z):
    """Points in the obstacle height band (cheap; first, so that the body test sees no floor)."""
    if points is None or len(points) == 0:
        return points
    p = np.asarray(points, np.float32)
    return p[(p[:, 2] > z_range[0]) & (p[:, 2] < z_range[1])]


def remove_body(points, body):
    """Drop points inside the body capsules (only points in the capsules' bounding box are tested)."""
    if points is None or len(points) == 0 or body is None:
        return points
    a, b, r = body
    p = np.asarray(points, np.float32)
    lo = np.minimum(a, b).min(0) - r.max(); hi = np.maximum(a, b).max(0) + r.max()
    near = np.flatnonzero(((p >= lo) & (p <= hi)).all(1))
    if len(near):
        q = p[near]
        ab = (b - a).astype(np.float32); L2 = np.maximum((ab * ab).sum(1), 1e-12)
        t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(-1) / L2, 0., 1.)
        d = np.linalg.norm(q[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1)
        keep = np.ones(len(p), bool); keep[near[(d < r[None]).any(1)]] = False
        p = p[keep]
    return p


# ------------------------------------------------------------------------------------------------ dynamic layer
def dynamic_voxels(points, first, shape, base_sdf=None, static_match=STATIC_MATCH_M, z_range=REAL_Z, min_block=2):
    """Window voxel indices [M, 3] of the points that the base layer does not explain."""
    if points is None or len(points) == 0:
        return np.zeros((0, 3), int)
    p = np.asarray(points, float)
    p = p[(p[:, 2] > z_range[0]) & (p[:, 2] < z_range[1])]
    idx = np.floor((p - (np.asarray(first) - DX / 2)) / DX).astype(int)
    idx = idx[(idx >= 0).all(1) & (idx < np.asarray(shape)).all(1)]
    if len(idx) == 0:
        return idx.reshape(0, 3)
    idx = np.unique(idx, axis=0)
    if base_sdf is not None:
        idx = idx[base_sdf[tuple(idx.T)] > static_match]
    if min_block > 1 and len(idx):                      # speckle: a lone voxel in its 12 cm block is lidar noise
        _, inv, cnt = np.unique(idx // 3, axis=0, return_inverse=True, return_counts=True)
        idx = idx[cnt[inv.ravel()] >= min_block]
    return idx


_KERNELS = {}


def _kernel(radius, dx):
    key = (round(radius, 4), round(dx, 6))
    if key not in _KERNELS:
        n = int(math.ceil(radius / dx)); o = np.arange(-n, n + 1)
        d = np.sqrt(o[:, None, None] ** 2 + o[None, :, None] ** 2 + o[None, None, :] ** 2) * dx - dx / 2
        _KERNELS[key] = (n, np.where(d <= radius, d, np.inf).astype(np.float32))
    return _KERNELS[key]


def stamp_sdf(idx, shape, dx=DX, radius=DYN_RADIUS_M):
    """Signed distance of the voxels idx, truncated: +inf farther than `radius`, -dx/2 in an occupied voxel (the
    zero level on voxel faces, as skfmm on a voxel occupancy)."""
    out = np.full(shape, np.inf, np.float32)
    if len(idx) == 0:
        return out
    centres = idx
    if len(idx) > STAMP_MAX_VOXELS:                                     # 8 cm centres: <= 2.8 cm extra error
        _, first = np.unique(idx // 2, axis=0, return_index=True); centres = idx[first]
    n, k = _kernel(radius, dx); S = np.asarray(shape)
    for c in centres:
        lo = c - n; hi = c + n + 1
        a = np.maximum(lo, 0); b = np.minimum(hi, S)
        dst = out[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
        np.minimum(dst, k[a[0] - lo[0]:b[0] - lo[0], a[1] - lo[1]:b[1] - lo[1], a[2] - lo[2]:b[2] - lo[2]], out=dst)
    out[tuple(idx.T)] = -dx / 2
    return out


def combine(base_sdf, idx, dx=DX):
    """Window sdf = min(base, dynamic) and its occupancy."""
    sdf = base_sdf if len(idx) == 0 else np.minimum(base_sdf, stamp_sdf(idx, base_sdf.shape, dx))
    return sdf, sdf < 0


def influence_box(idx, shape, radius=DYN_RADIUS_M, dx=DX):
    """[lo, hi) voxel box outside which the dynamic voxels change nothing (stamp radius + the gradient stencil), or None."""
    if len(idx) == 0:
        return None
    m = int(math.ceil(radius / dx)) + 2
    return np.maximum(idx.min(0) - m, 0), np.minimum(idx.max(0) + 1 + m, np.asarray(shape))


def union_box(*boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)


def main():
    ap = argparse.ArgumentParser(description='static ESDF layer from a map')
    ap.add_argument('pcd'); ap.add_argument('out')
    ap.add_argument('--path', help="mapping trajectory csv: points near it are cleared (robot body, gantry rope)")
    ap.add_argument('--extent', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                    help='default: the trajectory (or the origin) +- 7 m')
    ap.add_argument('--clear-radius', type=float, default=.35)
    a = ap.parse_args()
    import time
    pts = read_pcd(a.pcd); path_xy = read_path_xy(a.path) if a.path else None
    if a.extent:
        extent = a.extent
    else:
        c = path_xy if path_xy is not None and len(path_xy) else np.zeros((1, 2))
        extent = (c[:, 0].min() - 7, c[:, 0].max() + 7, c[:, 1].min() - 7, c[:, 1].max() + 7)
    t = time.time()
    s = build_static(pts, extent, path_xy, a.clear_radius)
    np.savez_compressed(a.out, sdf=s['sdf'], first=s['first'], dx=s['dx'])
    print('%s: %d map points, grid %s at %.0f cm over x %.1f..%.1f y %.1f..%.1f, %d occupied voxels, %.1f s'
          % (a.out, len(pts), s['sdf'].shape, DX * 100, *extent, s['occupied'], time.time() - t))


if __name__ == '__main__':
    main()
