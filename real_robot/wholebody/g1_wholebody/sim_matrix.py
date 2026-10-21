#!/usr/bin/env python3
"""Closed-loop matrix of the ROBOT executor (sim_loop.run) under robot-like conditions, gantry procedure.

  python -m g1_wholebody.sim_matrix --out results/closed_loop [--jobs 10]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SCENES = ['flat_walk_4m', 'sidegap_plain_047', 'sidegap_corridor_048', 'gap_044_exact']
ROBOT = dict(pose_hz=10., pose_latency_s=0.06, pose_noise=0.01, action_latency_s=0.010, field_latency_s=2.5, motor_kp_scale=0.9)
SETS = {
    'robot_like': [dict(ROBOT, policy=p) for p in ('robust_delay_v1_u150', 'u596')],
    'mount_mismatch': [dict(ROBOT, policy='robust_delay_v1_u150', true_mount='robot_xml', exec_mount='unitree_urdf')],
    'action_20ms': [dict(ROBOT, policy='robust_delay_v1_u150', action_latency_s=0.020)],
}
HOLD_DRIFT_M = 0.30        # 'balance' stands in place (the stop phase steps ~0.15 m); more between 'balance' and 'go' is a failure
SEEDS = {'robot_like': (0, 1, 2), 'mount_mismatch': (0, 1), 'action_20ms': (0, 1)}


def one(job):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from g1_wholebody.sim_loop import run
    name, kw, scene, seed = job
    r = run(scene=scene, seed=seed, procedure='gantry', seconds=45., **kw)
    r['set'] = name
    return r


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', default='results/closed_loop'); p.add_argument('--jobs', type=int, default=10)
    a = p.parse_args(); out = Path(a.out); (out / 'runs').mkdir(parents=True, exist_ok=True)
    jobs = [(name, kw, s, seed) for name, variants in SETS.items() for kw in variants for s in SCENES for seed in SEEDS[name]]
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(one, jobs))
    table = {}
    for r in results:
        key = '%s / %s' % (r['set'], r['policy']); cell = table.setdefault(key, {}).setdefault(r['scene'], dict(runs=0, reached=0, falls=0, damp=0, intrusion_runs=0, max_step_ms=0., max_hold_drift_m=0.))
        cell['runs'] += 1; cell['reached'] += r['reached']; cell['falls'] += r['fell']; cell['damp'] += r['final_mode'] == 'DAMP'
        cell['intrusion_runs'] += r['site_intrusion_steps'] > 0; cell['max_step_ms'] = max(cell['max_step_ms'], r['max_step_ms'])
        cell['max_hold_drift_m'] = max(cell['max_hold_drift_m'], r['hold_drift_m'])
        (out / 'runs' / ('%s_%s_%s_s%d.json' % (r['set'], r['policy'], r['scene'], r['settings']['seed']))).write_text(json.dumps(r, indent=1))
    lines = ['| set / policy | ' + ' | '.join(SCENES) + ' |', '|---' * (len(SCENES) + 1) + '|']
    for key, row in table.items():
        lines.append('| %s | ' % key + ' | '.join('%d/%d, %d falls, %d damp, %d intrusion, hold drift ≤ %.2f m' % (c['reached'], c['runs'], c['falls'], c['damp'], c['intrusion_runs'], c['max_hold_drift_m'])
                                                  for c in (row[s] for s in SCENES)) + ' |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n'); (out / 'summary.json').write_text(json.dumps(table, indent=1))
    print('\n'.join(lines))
    for r in results:
        if r['final_mode'] == 'DAMP' or r['fell'] or not r['reached'] or r['hold_drift_m'] > HOLD_DRIFT_M:
            print('  !', r['set'], r['policy'], r['scene'], r['settings']['seed'], r['final_mode'], r['final_reason'],
                  'reached' if r['reached'] else 'not reached', 'hold drift %.2f m' % r['hold_drift_m'])


if __name__ == '__main__':
    main()
