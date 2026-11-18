#!/usr/bin/env python3
"""The robot's own surface for the lidar self-filter: points on the visual meshes of cat/wholebody/robot.xml (the
training model), ~1 cm apart, in each body's frame -> g1_wholebody/body_points.npz (keys: names, points, body).

  python3 make_body_points.py <g1_description/meshes dir> [--spacing 0.01]

Run once when robot.xml changes (needs mujoco and the meshes; the robot only needs the npz).
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XML = HERE.parents[2] / 'cat' / 'wholebody' / 'robot.xml'
OUT = HERE.parent / 'g1_wholebody' / 'body_points.npz'


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sample_mesh(vert, face, spacing, rng):
    a, b, c = vert[face[:, 0]], vert[face[:, 1]], vert[face[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = max(int(area.sum() / spacing ** 2), len(face) // 4)
    tri = rng.choice(len(face), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n); flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])
    pts = np.vstack([pts, vert])                                       # the vertices too (edges, small parts)
    _, keep = np.unique(np.floor(pts / (spacing / 2)).astype(np.int64), axis=0, return_index=True)
    return pts[keep]


def main():
    import mujoco
    ap = argparse.ArgumentParser(); ap.add_argument('meshes'); ap.add_argument('--spacing', type=float, default=0.01)
    a = ap.parse_args()
    xml = XML.read_text().replace('file="assets/', 'file="%s/' % Path(a.meshes).resolve())
    model = mujoco.MjModel.from_xml_string(xml)
    rng = np.random.default_rng(0)
    per_body = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        m = model.geom_dataid[g]
        vert = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]].astype(float)
        face = model.mesh_face[model.mesh_faceadr[m]:model.mesh_faceadr[m] + model.mesh_facenum[m]]
        pts = sample_mesh(vert, face, a.spacing, rng) @ quat_matrix(model.geom_quat[g]).T + model.geom_pos[g]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        per_body.setdefault(name, []).append(pts)
    names = sorted(per_body)
    points = [np.vstack(per_body[n]).astype(np.float32) for n in names]
    np.savez_compressed(OUT, names=np.array(names), points=np.vstack(points),
                        body=np.concatenate([np.full(len(p), i, np.int16) for i, p in enumerate(points)]))
    print('%s: %d bodies, %d points at %.0f mm' % (OUT, len(names), sum(map(len, points)), a.spacing * 1000))
    for n, p in zip(names, points):
        print('  %-28s %6d pts, extent %s' % (n, len(p), np.ptp(p, 0).round(3)))


if __name__ == '__main__':
    main()
