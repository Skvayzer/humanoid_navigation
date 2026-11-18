#!/usr/bin/env python3
"""What each self-filter treats as the robot: the removed volume round the robot in one pose (side and front
silhouettes, and a slice through the right arm). python3 plot_self_filter_volume.py <out.png>"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from g1_wholebody import esdf_layers as L
from wholebody_core import load_contract, JOINT_NAMES

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#2b2b29', '#8a8984', '#e6e5e0'
CAPS, MESH, BODY = '#eb6834', '#2a78d6', '#7d7c76'

c = load_contract(); q = np.asarray(c['nominal'], float).copy()
q[JOINT_NAMES.index('right_shoulder_pitch_joint')] -= .9          # right arm reaching forward
q[JOINT_NAMES.index('right_elbow_joint')] -= .3
pos, quat = np.array([0., 0., .78]), np.array([1., 0., 0., 0.])
surf = L.body_surface(); kin = surf.kin
h = 0.015
xs, ys, zs = np.arange(-.45, .75, h), np.arange(-.55, .55, h), np.arange(.1, 1.4, h)
G = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1).reshape(-1, 3)
ca, cb, cr = L.body_capsules(kin, q, pos, quat)                    # the last capsule is the rope column: not shown
caps_keep = L.remove_body(np.column_stack([G, np.arange(len(G))]), (ca[:-1], cb[:-1], cr[:-1]))[:, 3].astype(int)
caps = np.ones(len(G), bool); caps[caps_keep] = False
mesh = surf.inside(G, dict(t=[0.], q=[q], pos=[pos], quat=[quat]))
shape = (len(xs), len(ys), len(zs)); caps, mesh = caps.reshape(shape), mesh.reshape(shape)
frames, palms = surf.posed(q, pos, quat)
d = np.load(L.BODY_POINTS); names = [str(n) for n in d['names']]
body = np.vstack([d['points'][d['body'] == i][::3].astype(float) @ frames[n][:3, :3].T + frames[n][:3, 3]
                  for i, n in enumerate(names) if n in frames])
palm = palms[1]; j = int(np.argmin(np.abs(ys - palm[1])))

fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.1, 1, 1.1]))
views = [('Side silhouette (projected along y)', caps.any(1), mesh.any(1), (xs, zs), (0, 2), None),
         ('Front silhouette (projected along x)', caps.any(0), mesh.any(0), (ys, zs), (1, 2), None),
         ('Slice through the right arm (y = %.2f m)' % ys[j], caps[:, j], mesh[:, j], (xs, zs), (0, 2), (ys[j], .03))]
for ax, (title, cv, mv, (a, b), (ia, ib), sl) in zip(axes, views):
    ax.set_facecolor(SURFACE)
    ext = (a[0] - h / 2, a[-1] + h / 2, b[0] - h / 2, b[-1] + h / 2)
    ax.imshow(np.where(cv.T, 1., np.nan), origin='lower', extent=ext, cmap=matplotlib.colors.ListedColormap([CAPS]), alpha=.28,
              interpolation='nearest')
    ax.imshow(np.where(mv.T, 1., np.nan), origin='lower', extent=ext, cmap=matplotlib.colors.ListedColormap([MESH]), alpha=.55,
              interpolation='nearest')
    pts = body if sl is None else body[np.abs(body[:, 1] - sl[0]) < sl[1]]
    ax.scatter(pts[:, ia], pts[:, ib], s=.6, c=BODY, lw=0)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect('equal')
    ax.set_title(title, color=INK, fontsize=11, loc='left', pad=8); ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel('xyz'[ia] + ' [m]', color=MUTED, fontsize=9); ax.set_ylabel('z [m]', color=MUTED, fontsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
handles = [plt.Rectangle((0, 0), 1, 1, color=CAPS, alpha=.4), plt.Rectangle((0, 0), 1, 1, color=MESH, alpha=.7),
           plt.Line2D([], [], ls='', marker='o', ms=3, color=BODY)]
labels = ['capsules: 12 cm round limb segments, 20 cm round the trunk (volume %.0f L)' % (caps.sum() * h ** 3 * 1000),
          'meshes: 4 cm round the robot\'s surface (volume %.0f L)' % (mesh.sum() * h ** 3 * 1000), 'robot surface (model)']
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=10, labelcolor=INK, bbox_to_anchor=(.5, 0.))
fig.suptitle('What each self-filter treats as the robot (one pose, right arm reaching forward): anything lidar sees in the '
             'shaded volume is dropped', color=INK, fontsize=13, x=.01, ha='left')
fig.text(.01, .925, 'Not shown: the gantry-rope column (30 cm round the pelvis, above 1 m), the same in both filters.',
         color=MUTED, fontsize=9.5)
fig.tight_layout(rect=(0, .10, 1, .91)); fig.savefig(sys.argv[1], dpi=150, facecolor=SURFACE)
print('capsule volume %.0f L, mesh volume %.0f L' % (caps.sum() * h ** 3 * 1000, mesh.sum() * h ** 3 * 1000))
