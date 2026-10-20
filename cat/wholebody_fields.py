"""Occupancy -> potential fields in the convention the whole-body policy was TRAINED with.

Port of the training generator (Click-and-Traverse-Mjlab cat_ppo/furniture/generalist_fields.make_clutter_fields ->
upstream CAT pf_modular.make_sdf / grad3 / make_guidance_field_progressive, Apache-2.0, see upstream/NOTICE.md).
Differences from field_core.py (the preview port of the older pf.py) that the policy can see:
  * inside obstacles the guidance is the outward boundary normal (field_core: zero) -- trilinear sampling next to a
    wall mixes those cells in, which is where hands and elbows are;
  * guidance and boundary are stored as UNIT directions (training packed them; magnitudes, incl. the 0.6 speed and
    its slow-down near the goal, never reached the policy);
  * values live on voxel CENTERS, returned as a FieldGrid whose grid points are those centers.
Unknown space must be passed as occupied (conservative, as field_core): training never saw unknown cells.
"""
import numpy as np
import skfmm

from wholebody_core import FieldGrid

R_PROJ_CELLS = 5.          # tangential blending radius (pf_modular default 5 voxels)
GOAL_SEED_RADIUS = .12     # pf_modular PFConfig.goal_seed_r


def training_fields(blocked, first_center, dx, goal):
    """blocked: bool [X,Y,Z] (occupied or unknown); first_center: world position of voxel (0,0,0)'s CENTER;
    goal: world (x, y, z) -- training used z = 0.75 m. Returns FieldGrid on voxel centers."""
    blocked = np.asarray(blocked, dtype=bool)
    origin = np.asarray(first_center, dtype=float); goal = np.asarray(goal, dtype=float)
    if blocked.ndim != 3 or min(blocked.shape) < 2 or not np.isfinite(origin).all() or not np.isfinite(goal).all():
        raise ValueError("invalid field geometry")
    if blocked.all():
        raise ValueError("no free space")
    eps = 1e-9
    if blocked.any():
        sdf = skfmm.distance(np.where(blocked, -1., 1.), dx=dx).astype(np.float32)
    else:
        sdf = np.full(blocked.shape, float(np.linalg.norm(np.array(blocked.shape)*dx)), np.float32)
    bf = np.stack(np.gradient(sdf, dx, dx, dx, edge_order=2), axis=-1).astype(np.float32)
    axes = [origin[i]+np.arange(blocked.shape[i])*dx for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    seed = ((X-goal[0])**2+(Y-goal[1])**2+(Z-goal[2])**2) <= GOAL_SEED_RADIUS**2
    if not (seed & ~blocked).any():
        raise ValueError("goal seed has no free cell (goal occupied, unknown or outside the volume)")
    phi = np.ma.MaskedArray(np.where(seed, -1., 1.), mask=blocked)
    T_ma = skfmm.distance(phi, dx=dx)
    free = ~np.ma.getmaskarray(T_ma)
    T = T_ma.filled(float(np.max(T_ma[free])) if free.any() else 0.).astype(np.float32)
    g = -np.stack(np.gradient(T, dx, dx, dx, edge_order=2), axis=-1).astype(np.float32)
    bnorm = np.linalg.norm(bf, axis=-1, keepdims=True)
    bunit = np.where(bnorm > eps, bf/np.maximum(bnorm, eps), 0.).astype(np.float32)
    g_perp = g-np.sum(g*bunit, axis=-1, keepdims=True)*bunit
    t = np.clip(np.maximum(sdf, 0.)/(R_PROJ_CELLS*dx+eps), 0., 1.)
    w = (1.-t*t*(3.-2.*t))[..., None]
    g_mix = (1.-w)*g+w*g_perp
    g_mix[blocked] = bunit[blocked]
    mag = np.linalg.norm(g_mix, axis=-1, keepdims=True)
    gf = np.where(mag > eps, g_mix/np.maximum(mag, eps), 0.).astype(np.float32)
    for a in (sdf, gf, bunit):
        if not np.isfinite(a).all():
            raise ValueError("nonfinite potential field")
    return FieldGrid(sdf, gf, bunit, origin, dx)
