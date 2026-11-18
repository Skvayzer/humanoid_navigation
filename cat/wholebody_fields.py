"""Occupancy -> potential fields in the convention the whole-body policy was TRAINED with.

Port of the training generator (Click-and-Traverse-Mjlab cat_ppo/furniture/generalist_fields.make_clutter_fields ->
upstream CAT pf_modular.make_sdf / grad3 / make_guidance_field_progressive, Apache-2.0, see upstream/NOTICE.md).
Differences from field_core.py (the preview port of the older pf.py) that the policy can see:
  * inside obstacles the guidance is the outward boundary normal (field_core: zero) -- trilinear sampling next to a
    wall mixes those cells in, which is where hands and elbows are;
  * guidance carries training's speed profile (pf_modular.make_guidance_field_progressive): 0.6 farther than 0.3 m
    (horizontally) from the goal, 0.6 (d/0.3)^3 inside. The policy's observations normalise it, but the stop decision
    (wholebody_core.command_from_fields on the undelayed fields, < 0.2 -> stop latch) uses the magnitude: training
    stops ~0.26 m from the goal. Before 2026-11-17 this port stored unit vectors and the robot stopped only inside the
    0.12 m goal seed. Boundary normals are stored as unit directions;
  * values live on voxel CENTERS, returned as a FieldGrid whose grid points are those centers.
Unknown space must be passed as occupied (conservative, as field_core): training never saw unknown cells.

Goal seed. goal_seed='ball' (every policy trained before 2026-11-18): the arrival-time map starts from a 12 cm ball at
the goal point, z = 0.75 m, so every guidance arrow points at that ball -- the head's tilts down and the feet's up by
angles that only depend on the distance to the goal (head -17 / feet +24 deg at 1.5 m, -50 / +60 deg at 0.4 m on open
floor). goal_seed='column' projects the goal onto the ground: the seed is the vertical column within 12 cm of the goal
xy over the perceived heights (0.10-1.40 m), so near the goal the arrows stay level at every height while obstacles
still bend them up or down. Use the seed the policy was trained with (policy_contract.json "goal_seed").
"""
import numpy as np
import skfmm

from wholebody_core import FieldGrid

R_PROJ_CELLS = 5.          # tangential blending radius (pf_modular default 5 voxels)
GOAL_SEED_RADIUS = .12     # pf_modular PFConfig.goal_seed_r
V_MAX, SLOW_RADIUS, SLOW_POWER = .6, .3, 3.   # pf_modular: PFConfig.v_max, T_thresh, p (the speed profile)
GOAL_SEEDS = ('ball', 'column')
COLUMN_Z = (.10, 1.40)     # column goal: the heights the robot perceives (robot perception volume)


def training_fields(blocked, first_center, dx, goal, goal_seed='ball'):
    """blocked: bool [X,Y,Z] (occupied or unknown); first_center: world position of voxel (0,0,0)'s CENTER;
    goal: world (x, y, z) -- training used z = 0.75 m (the column seed uses only x, y). Returns FieldGrid on voxel
    centers."""
    if goal_seed not in GOAL_SEEDS:
        raise ValueError(f"goal_seed must be one of {GOAL_SEEDS}, got {goal_seed!r}")
    blocked = np.asarray(blocked, dtype=bool)
    origin = np.asarray(first_center, dtype=float); goal = np.asarray(goal, dtype=float)
    if blocked.ndim != 3 or min(blocked.shape) < 2 or not np.isfinite(origin).all() or not np.isfinite(goal).all():
        raise ValueError("invalid field geometry")
    if blocked.all():
        raise ValueError("no free space")
    if blocked.any():
        sdf = skfmm.distance(np.where(blocked, -1., 1.), dx=dx).astype(np.float32)
    else:
        sdf = np.full(blocked.shape, float(np.linalg.norm(np.array(blocked.shape)*dx)), np.float32)
    g = goal_gradient(blocked, origin, dx, goal, goal_seed)
    gf, bunit = mix_guidance(sdf, g, blocked, dx, origin, goal)
    return FieldGrid(sdf, gf, bunit, origin, dx)


def goal_gradient(blocked, first_center, dx, goal, goal_seed='ball'):
    """Raw guidance before the boundary blending: -grad of the geodesic distance to the goal seed (the slow part).
    Depends on the occupancy, the goal and the goal seed only, so it can be reused while the sdf near it is refreshed."""
    if goal_seed not in GOAL_SEEDS:
        raise ValueError(f"goal_seed must be one of {GOAL_SEEDS}, got {goal_seed!r}")
    origin = np.asarray(first_center, dtype=float); goal = np.asarray(goal, dtype=float)
    axes = [origin[i]+np.arange(blocked.shape[i])*dx for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    if goal_seed == 'ball':
        seed = ((X-goal[0])**2+(Y-goal[1])**2+(Z-goal[2])**2) <= GOAL_SEED_RADIUS**2
    else:
        seed = (((X-goal[0])**2+(Y-goal[1])**2) <= GOAL_SEED_RADIUS**2) & (Z >= COLUMN_Z[0]) & (Z <= COLUMN_Z[1])
    if not (seed & ~blocked).any():
        raise ValueError("goal seed has no free cell (goal occupied, unknown or outside the volume)")
    phi = np.ma.MaskedArray(np.where(seed, -1., 1.), mask=blocked)
    T_ma = np.ma.asarray(skfmm.distance(phi, dx=dx))     # skfmm returns a plain array when nothing is masked
    free = ~np.ma.getmaskarray(T_ma)
    T = T_ma.filled(float(np.max(T_ma[free])) if free.any() else 0.).astype(np.float32)
    return -np.stack(np.gradient(T, dx, dx, dx, edge_order=2), axis=-1).astype(np.float32)


def speed_profile(first_center, shape, dx, goal):
    """Training's guidance magnitude on the voxel centres: V_MAX, slowing as (d / SLOW_RADIUS)^SLOW_POWER inside
    SLOW_RADIUS of the goal (d: horizontal distance), shape [X, Y, 1, 1] (it does not depend on z)."""
    x = first_center[0] + np.arange(shape[0])*dx - goal[0]
    y = first_center[1] + np.arange(shape[1])*dx - goal[1]
    d = np.hypot(x[:, None], y[None, :])
    return np.where(d > SLOW_RADIUS, V_MAX, V_MAX*(d/SLOW_RADIUS)**SLOW_POWER).astype(np.float32)[:, :, None, None]


def mix_guidance(sdf, g, blocked, dx, first_center=None, goal=None):
    """sdf + raw guidance -> (guidance gf, unit boundary bf), as training: guidance turned tangential within
    R_PROJ_CELLS of a wall, the outward normal inside obstacles, then scaled by the speed profile (with a goal; without
    one: unit directions). first_center: world position of sdf[0, 0, 0]'s centre."""
    eps = 1e-9
    bf = np.stack(np.gradient(sdf, dx, dx, dx, edge_order=2), axis=-1).astype(np.float32)
    bnorm = np.linalg.norm(bf, axis=-1, keepdims=True)
    bunit = np.where(bnorm > eps, bf/np.maximum(bnorm, eps), 0.).astype(np.float32)
    g_perp = g-np.sum(g*bunit, axis=-1, keepdims=True)*bunit
    t = np.clip(np.maximum(sdf, 0.)/(R_PROJ_CELLS*dx+eps), 0., 1.)
    w = (1.-t*t*(3.-2.*t))[..., None]
    g_mix = (1.-w)*g+w*g_perp
    g_mix[blocked] = bunit[blocked]
    mag = np.linalg.norm(g_mix, axis=-1, keepdims=True)
    gf = np.where(mag > eps, g_mix/np.maximum(mag, eps), 0.).astype(np.float32)
    if goal is not None:
        gf *= speed_profile(np.asarray(first_center, float), sdf.shape, dx, np.asarray(goal, float))
    return gf, bunit
