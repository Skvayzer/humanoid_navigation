"""Levels a saved .vmap (FAST_LIO's voxel map): rotates the whole map about its origin so that its floor is flat.

    python scripts/vmap_level.py <map.vmap> <out.vmap> [--tilt-only]

The floor: the planar voxels whose normal is within 10 deg of z, in the lowest dense layer below the origin (the
mapping sensor), fitted as one plane through their means (weighted by points, outliers trimmed). The rotation is the
smallest one taking the floor's normal onto z (the heading is kept). It is applied as VoxelMap::save applies its
level: each voxel's statistics, rotated exactly, go into the voxel of its rotated centre (voxels meeting there are
merged). The header (frame, parameters) is kept. --tilt-only prints the floor's tilt and writes nothing.
The .vmap format (version 3, little-endian here) is described in include/voxel_map.h.
"""
import argparse
import struct
import sys
import numpy as np

HEAD = struct.Struct('<4sII3d4d7dQ')     # magic, version, byte-order mark, frame t, q, parameters, voxel count
VOXEL = struct.Struct('<Qidd3d9d')       # key, n, w, w2, sum, sum_sq (relative to the voxel's corner)
OFF = 1 << 20


def unpack(key):
    return np.array([((key >> 42) & 0x1FFFFF) - OFF, ((key >> 21) & 0x1FFFFF) - OFF, (key & 0x1FFFFF) - OFF])


def pack(idx):
    x, y, z = (int(i) + OFF for i in idx)
    return ((x & 0x1FFFFF) << 42) | ((y & 0x1FFFFF) << 21) | (z & 0x1FFFFF)


def read(path):
    data = open(path, 'rb').read()
    head = HEAD.unpack_from(data, 0)
    if head[0] != b'VXMP' or head[1] != 3 or head[2] != 0x01020304:
        sys.exit('%s: not a version-3 .vmap of this byte order' % path)
    size, n_voxels = head[10], head[17]
    voxels = []
    for i in range(n_voxels):
        r = VOXEL.unpack_from(data, HEAD.size + i * VOXEL.size)
        voxels.append((unpack(r[0]) * size, r[1], r[2], r[3], np.array(r[4:7]), np.array(r[7:16]).reshape(3, 3)))
    return head, voxels


def floor_normal(head, voxels):
    size, point_sigma = head[10], head[11]
    means, normals, counts = [], [], []
    for origin, n, w, w2, s, ss in voxels:
        m = s / w
        cov = ss / w - np.outer(m, m) + point_sigma ** 2 * w2 / (w * w) * np.eye(3)
        l, e = np.linalg.eigh(cov)
        if l[0] <= 0.1 * l[1] and abs(e[2, 0]) > np.cos(np.radians(10)):
            means.append(origin + m), normals.append(e[:, 0]), counts.append(n)
    P, c = np.array(means), np.array(counts, float)
    below = P[:, 2] < -0.2
    P, c = P[below], c[below]
    hist, edges = np.histogram(P[:, 2], bins=np.arange(P[:, 2].min(), -0.2 + 0.05, 0.05), weights=c)
    ok = hist >= 0.3 * hist.max()   # the lowest bin with a good share of the strongest one: the floor, not a table
    z0 = edges[np.argmax(ok)] + 0.025
    sel = np.abs(P[:, 2] - z0) < 0.3
    P, c = P[sel], c[sel]
    for _ in range(5):
        centre = (c[:, None] * P).sum(0) / c.sum()
        _, _, vt = np.linalg.svd(np.sqrt(c)[:, None] * (P - centre), full_matrices=False)
        n = vt[2] * np.sign(vt[2][2])
        r = (P - centre) @ n
        keep = np.abs(r) < max(3 * np.sqrt((c * r * r).sum() / c.sum()), 0.03)
        P, c = P[keep], c[keep]
    return n, centre, len(P), np.sqrt((c * r[keep] ** 2).sum() / c.sum())


def rotation_onto_z(n):
    v, s, cth = np.cross(n, [0, 0, 1]), np.linalg.norm(np.cross(n, [0, 0, 1])), n[2]
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - cth) / s ** 2)


def rotate(voxels, size, R):
    out = {}
    for origin, n, w, w2, s, ss in voxels:
        o, sr = R @ origin, R @ s
        idx = np.floor(R @ (origin + size / 2) / size).astype(int)
        key = pack(idx)
        r = out.setdefault(key, [idx * size, 0, 0.0, 0.0, np.zeros(3), np.zeros((3, 3))])
        d = o - r[0]   # q' = R q + d: sums relative to the new voxel's corner
        r[1] += n
        r[2] += w
        r[3] += w2
        r[4] = r[4] + sr + w * d
        r[5] = r[5] + R @ ss @ R.T + np.outer(sr, d) + np.outer(d, sr) + w * np.outer(d, d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('vmap')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--tilt-only', action='store_true')
    args = ap.parse_args()
    head, voxels = read(args.vmap)
    n, centre, count, rms = floor_normal(head, voxels)
    print('%s: floor tilted %.3f deg (normal %s, %d voxels at z %.2f m, rms %.1f cm)' % (
        args.vmap, np.degrees(np.arccos(n[2])), np.round(n, 5), count, centre[2], rms * 100))
    if args.tilt_only:
        return
    if not args.out:
        sys.exit('an output path is needed')
    out = rotate(voxels, head[10], rotation_onto_z(n))
    with open(args.out, 'wb') as f:
        f.write(HEAD.pack(*head[:17], len(out)))
        for key, (origin, cnt, w, w2, s, ss) in out.items():
            f.write(VOXEL.pack(key, cnt, w, w2, *s, *ss.reshape(-1)))
    print('%s: %d voxels (from %d)' % (args.out, len(out), len(voxels)))


if __name__ == '__main__':
    main()
