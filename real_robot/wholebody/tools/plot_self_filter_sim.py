#!/usr/bin/env python3
"""Figures from self_filter_sim.py's output: capsules (before) vs the swept mesh filter (after), side view.

  python3 plot_self_filter_sim.py <self_filter_sim out dir>      -> before_after_walk.png, capsules_problem_stand.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import self_filter_sim as S                                       # noqa: E402

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#2b2b29', '#8a8984', '#e6e5e0'
KEPT, PHANTOM, LOST = '#2a78d6', '#eb6834', '#1baf7a'             # categorical slots 1-3 (validated, all pairs)
BODY = '#c9c8c2'


def window(pts, t_end):
    w = pts[(pts[:, 3] > t_end - .5) & (pts[:, 3] <= t_end)]
    w = w[np.argsort(-w[:, 3], kind='stable')]
    _, first = np.unique(np.floor(w[:, :3] / .04).astype(np.int64), axis=0, return_index=True); w = w[first]
    return w[(w[:, 2] > .10) & (w[:, 2] < 1.40)]


def filters(w, snaps_i, d):
    from g1_wholebody import esdf_layers as L
    i = snaps_i; lo = max(0, i - 11)
    hist = dict(t=d['t'][lo:i + 1], q=d['q'][lo:i + 1], pos=d['pos'][lo:i + 1], quat=d['quat'][lo:i + 1])
    q, pos, quat = d['q'][i], d['pos'][i], d['quat'][i]
    kin = L.body_surface().kin
    kept_idx = L.remove_body(np.column_stack([w[:, :3], np.arange(len(w))]), L.body_capsules(kin, q, pos, quat))[:, 3].astype(int)
    caps = np.ones(len(w), bool); caps[kept_idx] = False
    mesh = L.body_surface().inside(S.scan_time(w), hist, sweep=.05)
    return dict(capsules=caps, mesh=mesh), kin.world_sites(q, pos, quat), L.body_surface().posed(q, pos, quat)[0]


def body_cloud(frames, every=4):
    from g1_wholebody import esdf_layers as L
    d = np.load(L.BODY_POINTS); names = [str(n) for n in d['names']]; out = []
    for i, n in enumerate(names):
        if n in frames:
            p = d['points'][d['body'] == i][::every].astype(float); T = frames[n]; out.append(p @ T[:3, :3].T + T[:3, 3])
    return np.vstack(out)


def panel(ax, w, removed, sites, body, view, lim, title, note):
    lab = w[:, 4]; robot = lab == S.ROBOT; obstacle = (lab != S.ROBOT) & (lab != S.FLOOR)
    x, y = view
    ax.set_facecolor(SURFACE)
    ax.scatter(body[:, x], body[:, y], s=1.2, c=BODY, lw=0, zorder=1, label='robot (model)')
    ax.scatter(w[obstacle & ~removed, x], w[obstacle & ~removed, y], s=22, c=KEPT, marker='s', lw=0, zorder=3,
               label='obstacle voxel, kept')
    ax.scatter(w[obstacle & removed, x], w[obstacle & removed, y], s=34, c=LOST, marker='X', lw=0, zorder=4,
               label='obstacle voxel, wrongly removed')
    ax.scatter(w[robot & ~removed, x], w[robot & ~removed, y], s=40, c=PHANTOM, marker='^', lw=0, zorder=5,
               label='robot voxel left in (phantom obstacle)')
    ax.scatter(sites[:, x], sites[:, y], s=16, facecolor='white', edgecolor=INK, lw=1, zorder=6, label='policy body sites')
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect('equal')
    ax.set_title(title, color=INK, fontsize=12, loc='left', pad=22)
    ax.text(0, 1.015, note, transform=ax.transAxes, color=MUTED, fontsize=9.5, va='bottom')
    ax.grid(color=GRID, lw=.6); ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_xlabel('xyz'[x] + ' [m]', color=MUTED, fontsize=9); ax.set_ylabel('xyz'[y] + ' [m]', color=MUTED, fontsize=9)


def describe(w, removed, sites):
    lab = w[:, 4]; robot = lab == S.ROBOT; obstacle = (lab != S.ROBOT) & (lab != S.FLOOR)
    leak = robot & ~removed; lost = obstacle & removed
    near = lambda m: (np.min(np.linalg.norm(w[m][:, None, :3] - sites[None], axis=-1), axis=1) < .5).sum() if m.any() else 0
    parts = ['%d phantom voxel%s' % (leak.sum(), '' if leak.sum() == 1 else 's')]
    if leak.any():
        parts[-1] += ', nearest %.2f m from a body site' % np.min(np.linalg.norm(w[leak][:, None, :3] - sites[None], axis=-1))
    parts.append('%d of %d obstacle voxels near the body removed' % (near(lost), near(obstacle)))
    return ' · '.join(parts)


def figure(scenario, d, i, view, lim, suptitle, path):
    w = window(d['pts'], d['t'][i]); rem, sites, frames = filters(w, i, d); body = body_cloud(frames)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, (('capsules', 'Before: capsules, last pose'),
                                       ('mesh', 'After: meshes, each point\'s scan pose'))):
        panel(ax, w, rem[key], sites, body, view, lim, title, describe(w, rem[key], sites))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=9, labelcolor=INK, markerscale=1.3)
    fig.suptitle(suptitle, color=INK, fontsize=14, x=.02, ha='left', y=.985)
    fig.tight_layout(rect=(0, .07, 1, .93)); fig.savefig(path, dpi=160, facecolor=SURFACE); plt.close(fig)
    print('wrote', path)


CAPS_SHADE, MESH_SHADE = '#eb6834', '#2a78d6'                   # the filters' volumes, as in plot_self_filter_volume.py


def problem_figure(d, i, path):
    """The object at the right palm: each filter's removed volume (a 6 cm slab through the palm, side view) with the
    object's voxels in it, kept or removed."""
    from g1_wholebody import esdf_layers as L
    w = window(d['pts'], d['t'][i]); rem, sites, frames = filters(w, i, d)
    q, pos, quat = d['q'][i], d['pos'][i], d['quat'][i]
    surf = L.body_surface(); kin = surf.kin; palm = surf.posed(q, pos, quat)[1][1]
    half = .03; h = .01
    xs = np.arange(palm[0] - .45, palm[0] + .45, h); zs = np.arange(palm[2] - .4, palm[2] + .35, h)
    ys = np.arange(palm[1] - half, palm[1] + half + 1e-9, h)
    G = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1).reshape(-1, 3)
    ca, cb, cr = L.body_capsules(kin, q, pos, quat)                 # without the rope column (the same in both)
    keep = L.remove_body(np.column_stack([G, np.arange(len(G))]), (ca[:-1], cb[:-1], cr[:-1]))[:, 3].astype(int)
    vol = dict(capsules=np.ones(len(G), bool), mesh=surf.inside(G, dict(t=[0.], q=[q], pos=[pos], quat=[quat])))
    vol['capsules'][keep] = False
    shape = (len(xs), len(ys), len(zs))
    body = body_cloud(frames, every=2); body = body[np.abs(body[:, 1] - palm[1]) < half]
    lab = w[:, 4]; obj = (lab == S.MOVING) & (np.abs(w[:, 1] - palm[1]) < half + .02)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), facecolor=SURFACE)
    for ax, key, shade, title in ((axes[0], 'capsules', CAPS_SHADE, 'Before: capsules (12 cm round the arm)'),
                                  (axes[1], 'mesh', MESH_SHADE, 'After: meshes (4 cm round the surface)')):
        ax.set_facecolor(SURFACE)
        v = vol[key].reshape(shape).any(1)
        ax.imshow(np.where(v.T, 1., np.nan), origin='lower', extent=(xs[0] - h / 2, xs[-1] + h / 2, zs[0] - h / 2, zs[-1] + h / 2),
                  cmap=matplotlib.colors.ListedColormap([shade]), alpha=.3, interpolation='nearest', zorder=0)
        ax.scatter(body[:, 0], body[:, 2], s=1.5, c='#7d7c76', lw=0, zorder=1)
        k = obj & ~rem[key]; r = obj & rem[key]
        ax.scatter(w[k, 0], w[k, 2], s=46, marker='s', facecolor=INK, edgecolor='white', lw=1, zorder=3)
        ax.scatter(w[r, 0], w[r, 2], s=52, marker='X', facecolor='white', edgecolor=INK, lw=1.1, zorder=3)
        ax.scatter([palm[0]], [palm[2]], s=60, facecolor='white', edgecolor=INK, lw=1.4, zorder=4)
        ax.annotate('right palm', (palm[0], palm[2]), xytext=(-70, -32), textcoords='offset points', color=MUTED, fontsize=9,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(zs[0], zs[-1]); ax.set_aspect('equal')
        ax.set_title(title, color=INK, fontsize=12, loc='left', pad=22)
        ax.text(0, 1.015, '%d of %d voxels of the object removed (it vanishes for the policy)' % (r.sum(), obj.sum())
                if key == 'capsules' else '%d of %d voxels of the object removed (those within 4 cm of the hand and fingers)'
                % (r.sum(), obj.sum()), transform=ax.transAxes, color=MUTED, fontsize=9.5, va='bottom')
        ax.grid(color=GRID, lw=.6); ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_xlabel('x [m]', color=MUTED, fontsize=9); ax.set_ylabel('z [m]', color=MUTED, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAPS_SHADE, alpha=.45), plt.Rectangle((0, 0), 1, 1, color=MESH_SHADE, alpha=.45),
               plt.Line2D([], [], ls='', marker='s', ms=8, mfc=INK, mec='white'),
               plt.Line2D([], [], ls='', marker='X', ms=8, mfc='white', mec=INK),
               plt.Line2D([], [], ls='', marker='o', ms=3, color='#7d7c76')]
    labels = ['capsule volume (dropped as the robot)', 'mesh volume (dropped as the robot)', 'object voxel, kept: the policy sees it',
              'object voxel, removed', 'robot surface (model)']
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle('Problem case for the capsules: a 12 cm cube centred 8 cm in front of the right palm (standing, t = %.1f s)' % d['t'][i],
                 color=INK, fontsize=14, x=.02, ha='left', y=.985)
    fig.text(.02, .935, 'Side view of a 6 cm slab through the palm; shaded: the volume each filter treats as the robot.',
             color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, .07, 1, .92)); fig.savefig(path, dpi=160, facecolor=SURFACE); plt.close(fig)
    print('wrote', path)


def main():
    out = Path(sys.argv[1])
    walk = dict(np.load(out / 'walk_lidar.npz')); stand = dict(np.load(out / 'stand_lidar.npz'))
    rows = json.loads((out / 'walk_rows.json').read_text())
    caps = [r for r in rows if r['filter'] == 'capsules']
    worst = max(caps, key=lambda r: (r['leaked_within_0_3m_of_a_site'], r['leaked']))
    i = int(np.argmin(np.abs(walk['t'] - worst['t'])))
    p = walk['pos'][i]
    figure('walk', walk, i, (0, 1), ((p[0] - .85, p[0] + .85), (p[1] - .32, p[1] + .78)),
           'Lidar self-filter, simulated walk past a box (t = %.1f s): the same 0.5 s of lidar, top view' % worst['t'],
           out / 'before_after_walk.png')
    rows = json.loads((out / 'stand_rows.json').read_text())
    caps = [r for r in rows if r['filter'] == 'capsules' and 'approaching object_kept_near' in r and r['t'] > 16.]
    worst = min(caps, key=lambda r: r['approaching object_kept_near'])
    i = int(np.argmin(np.abs(stand['t'] - worst['t'])))
    problem_figure(stand, i, out / 'capsules_problem_stand.png')


if __name__ == '__main__':
    main()
