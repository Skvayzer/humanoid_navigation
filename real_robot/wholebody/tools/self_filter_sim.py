#!/usr/bin/env python3
"""Virtual test of the lidar self-filter: the policy in MuJoCo (sim_loop), a simulated head Mid-360 cast against the
robot's meshes and the scene, the real pipeline (0.5 s window, 4 cm voxels with the newest scan time, 0.10-1.40 m band,
blind 0.3 m) and three filters: the old capsules, the meshes posed at the end of the window, the meshes posed at each
point's scan time (the new default). Every ray knows what it hit, so the leaks and the losses are exact.

  python3 self_filter_sim.py --out <dir> [--scenarios walk stand]     (PYTHONPATH=real_robot/wholebody:cat)

walk:  fridge_side_050 (a 0.7 x 0.6 x 1.8 m box beside the goal): arms swing, the body passes the box.
stand: balance in place; an object (12 cm cube) approaches the right palm from 0.6 m to 8 cm and stays; another one is
       held 8 cm in front of the left palm the whole time (how each filter treats a held object).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MESHES = ROOT.parents[1] / 'data' / 'third_party' / 'unitree_ros' / 'robots' / 'g1_description' / 'meshes'
SCAN_HZ, BATCH_S, RAYS_PER_BATCH = 10., .01, 400          # Mid-360 ~200 k pts/s; keep a 1:5 share as FAST-LIO does
ELEV = (-7., 52.)                                          # Mid-360 vertical field of view [deg] in its own frame
BLIND, RANGE, NOISE = .3, 4., .02
GROUPS = np.array([1, 0, 1, 0, 0, 0], np.uint8)            # floor + added objects (0), the visual meshes (2); not collision
ROBOT, SCENE, MOVING, HELD, FLOOR = 0, 1, 2, 3, 4
LABELS = {ROBOT: 'robot', SCENE: 'scene box', MOVING: 'approaching object', HELD: 'held object', FLOOR: 'floor'}


def run_policy(scenario, record):
    from g1_wholebody.sim_loop import run
    if scenario == 'walk':
        return run('fridge_side_050', anchor=(0., 0., 0.), seconds=22., procedure='gantry', field_latency_s=0.9,
                   fence_margin=3., upper_filter='ema', upper_alpha=0.3, seed=1, stop_at_goal=False, record=record)
    return run('flat_walk_4m', anchor=(0., 0., 0.), seconds=22., procedure='gantry', field_latency_s=0.9, fence_margin=3.,
               upper_filter='ema', upper_alpha=0.3, seed=1, stop_at_goal=False, record=record,
               events=[(10.5, 'goal', (0.02, 0.0))])                 # 'go' at 11 s to here: it keeps standing


def lidar_model(scenario):
    """robot.xml with meshes + the scene's boxes + two mocap cubes (approaching, held)."""
    import mujoco
    from g1_wholebody.fields import load_scene
    xml = (ROOT / 'cat' / 'wholebody' / 'robot.xml').read_text().replace('file="assets/', 'file="%s/' % MESHES.resolve())
    extra = []
    if scenario == 'walk':
        for i, b in enumerate(load_scene('fridge_side_050')['boxes']):
            extra.append('<geom name="scene%d" type="box" pos="%s" size="%s" rgba=".8 .5 .2 1"/>'
                         % (i, ' '.join(map(str, b['center'])), ' '.join(map(str, b['half_size']))))
    else:
        for name in ('moving', 'held'):
            extra.append('<body name="%s" mocap="true" pos="5 5 1"><geom name="%s" type="box" size=".06 .06 .06"/></body>'
                         % (name, name))
    m = mujoco.MjModel.from_xml_string(xml.replace('</worldbody>', '\n'.join(extra) + '\n</worldbody>', 1))
    dbg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'debug_axis')     # robot.xml's own marker: not seen by the lidar
    if dbg >= 0:
        m.geom_group[m.geom_bodyid == dbg] = 5
    return m


def simulate_lidar(scenario, record, rng):
    import mujoco
    from g1_wholebody.frames import MOUNTS, rpy_matrix, transform
    m = lidar_model(scenario); d = mujoco.MjData(m)
    spec = MOUNTS['unitree_urdf']; t_torso_lidar = transform(rpy_matrix(*spec['rpy']), spec['xyz'])
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
    pelvis_root = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')]
    label_of_body = np.full(m.nbody, SCENE)
    label_of_body[m.body_rootid == pelvis_root] = ROBOT
    for name, lab in (('moving', MOVING), ('held', HELD)):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b >= 0:
            label_of_body[b] = lab
    floor = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    sites = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ('left_palm', 'right_palm')}
    mocap = {n: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)] for n in ('moving', 'held')
             if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
    times = np.array([r[0] for r in record]); qpos = np.array([r[1] for r in record])
    pts, snaps = [], []
    t = 2.0
    while t < times[-1]:
        k = int(np.searchsorted(times, t)); k = min(k, len(times) - 1)
        d.qpos[:qpos.shape[1]] = qpos[k]; mujoco.mj_kinematics(m, d)
        if scenario == 'stand':
            right, left = d.site_xpos[sites['right_palm']].copy(), d.site_xpos[sites['left_palm']].copy()
            a = np.clip((t - 13.) / 3., 0., 1.)                             # 13-16 s: from 0.6 m to 8 cm, then stays
            d.mocap_pos[mocap['moving']] = right + np.array([.08 + .52 * (1 - a), -.05, 0.])
            d.mocap_pos[mocap['held']] = left + np.array([.08, 0., 0.])
            mujoco.mj_kinematics(m, d)
        T = np.eye(4); T[:3, :3] = d.xmat[torso].reshape(3, 3); T[:3, 3] = d.xpos[torso]
        L = T @ t_torso_lidar
        az = rng.uniform(0, 2 * np.pi, RAYS_PER_BATCH); el = np.radians(rng.uniform(*ELEV, RAYS_PER_BATCH))
        v = np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]) @ L[:3, :3].T
        # the sensor sits inside the head shell (its window is transparent): each ray starts at the blind distance
        gid = np.full(RAYS_PER_BATCH, -1, np.int32); dist = np.full(RAYS_PER_BATCH, -1.); g1 = np.zeros(1, np.int32)
        for j in range(RAYS_PER_BATCH):
            dj = mujoco.mj_ray(m, d, L[:3, 3] + BLIND * v[j], v[j], GROUPS, 1, -1, g1)
            if dj >= 0 and BLIND + dj < RANGE:
                gid[j], dist[j] = g1[0], BLIND + dj
        ok = gid >= 0
        r = dist[ok] + rng.normal(0, NOISE, ok.sum())
        p = L[:3, 3] + v[ok] * r[:, None]
        lab = label_of_body[m.geom_bodyid[gid[ok]]]; lab[np.isin(gid[ok], floor)] = FLOOR
        pts.append(np.column_stack([p, np.full(len(p), t), lab]))
        if abs(t * SCAN_HZ - round(t * SCAN_HZ)) < 1e-6:                    # pose history at the lidar's rate
            snaps.append((t, qpos[k, 7:36].copy(), qpos[k, :3].copy(), qpos[k, 3:7].copy()))
        t = round(t + BATCH_S, 4)
    return np.vstack(pts), snaps


def evaluate(scenario, pts, snaps, out):
    from g1_wholebody import esdf_layers as L
    from g1_wholebody.kinematics import FastKinematics
    from wholebody_core import load_contract
    kin = FastKinematics(load_contract()); surf = L.body_surface(); surf6 = L.BodySurface(kin, margin=.06)
    snap_t = np.array([s[0] for s in snaps]); rows = []
    for i in range(5, len(snaps)):                                         # every scan, the last 0.5 s
        t_end = snap_t[i]
        w = pts[(pts[:, 3] > t_end - .5) & (pts[:, 3] <= t_end)]
        w = w[np.argsort(-w[:, 3], kind='stable')]
        _, first = np.unique(np.floor(w[:, :3] / .04).astype(np.int64), axis=0, return_index=True); w = w[first]
        w = w[(w[:, 2] > .10) & (w[:, 2] < 1.40)]
        _, q, pos, quat = snaps[i]
        hist = dict(t=snap_t[max(0, i - 11):i + 1], q=np.array([s[1] for s in snaps[max(0, i - 11):i + 1]]),
                    pos=np.array([s[2] for s in snaps[max(0, i - 11):i + 1]]),
                    quat=np.array([s[3] for s in snaps[max(0, i - 11):i + 1]]))
        last = {k: v[-1:] for k, v in hist.items()}
        site_pos = kin.world_sites(q, pos, quat)
        removed = {
            'none': np.zeros(len(w), bool),
            'capsules': ~np.isin(np.arange(len(w)), _kept_index(w[:, :3], L.remove_body(np.column_stack([w[:, :3], np.arange(len(w))]),
                                                                                       L.body_capsules(kin, q, pos, quat)))),
            'mesh, end of window': surf.inside(w[:, :3], last),
            'mesh, point time': surf.inside(w[:, :4], hist),
            # on the robot a point only has its scan's stamp (scan end): the scan centre = end - 50 ms
            'mesh, scan time': surf.inside(scan_time(w), hist),
            'mesh 6 cm, scan time': surf6.inside(scan_time(w), hist),
            'mesh, scan time, swept': surf.inside(scan_time(w), hist, sweep=.05),
        }
        dist_site = np.min(np.linalg.norm(w[:, None, :3] - site_pos[None], axis=-1), axis=1)
        for name, rem in removed.items():
            kept = ~rem; lab = w[:, 4]
            leak = kept & (lab == ROBOT)
            row = dict(t=float(t_end), filter=name, robot_voxels=int((lab == ROBOT).sum()), leaked=int(leak.sum()),
                       leaked_within_0_3m_of_a_site=int((leak & (dist_site < .3)).sum()),
                       phantom_min_dist_m=float(dist_site[leak].min()) if leak.any() else None)
            for l in (SCENE, MOVING, HELD):
                near = (lab == l) & (dist_site < .5)
                if near.any():
                    row['%s_kept_near' % LABELS[l]] = float(kept[near].mean())
            rows.append(row)
    (out / ('%s_rows.json' % scenario)).write_text(json.dumps(rows))
    return rows


def scan_time(w):
    t = np.ceil(w[:, 3] * SCAN_HZ - 1e-6) / SCAN_HZ - .5 / SCAN_HZ
    return np.column_stack([w[:, :3], t])


def _kept_index(xyz, kept):
    return kept[:, 3].astype(int) if len(kept) else np.zeros(0, int)


def summary(scenario, rows):
    print('\n%s: %d scans' % (scenario, len({r['t'] for r in rows})))
    keys = sorted({k for r in rows for k in r if k.endswith('_kept_near')})
    head = '%-22s %-14s %-12s %-22s %-14s' % ('filter', 'robot voxels', 'leaked', 'scans with a phantom', 'nearest phantom')
    print(head + ''.join(' %-22s' % k.replace('_kept_near', ' kept') for k in keys))
    for name in ('none', 'capsules', 'mesh, end of window', 'mesh, point time', 'mesh, scan time', 'mesh 6 cm, scan time', 'mesh, scan time, swept'):
        rs = [r for r in rows if r['filter'] == name]
        ph = [r['phantom_min_dist_m'] for r in rs if r['phantom_min_dist_m'] is not None]
        line = '%-22s %-14.0f %-12.1f %-22s %-14s' % (
            name, np.mean([r['robot_voxels'] for r in rs]), np.mean([r['leaked'] for r in rs]),
            '%.0f %% (< 0.3 m of a site)' % (100 * np.mean([r['leaked_within_0_3m_of_a_site'] > 0 for r in rs])),
            '%.2f m' % min(ph) if ph else '-')
        for k in keys:
            v = [r[k] for r in rs if k in r]; line += ' %-22s' % ('%.0f %%' % (100 * np.mean(v)) if v else '-')
        print(line)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--scenarios', nargs='+', default=['walk', 'stand'])
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for sc in a.scenarios:
        record = []; res = run_policy(sc, record)
        pts, snaps = simulate_lidar(sc, record, np.random.default_rng(0))
        np.savez_compressed(out / ('%s_lidar.npz' % sc), pts=pts, t=np.array([s[0] for s in snaps]),
                            q=np.array([s[1] for s in snaps]), pos=np.array([s[2] for s in snaps]),
                            quat=np.array([s[3] for s in snaps]))
        rows = evaluate(sc, pts, snaps, out); summary(sc, rows)
        print('  policy run: reached %s, fell %s, %d lidar points' % (res['reached'], res['fell'], len(pts)))


if __name__ == '__main__':
    main()
