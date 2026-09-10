"""CAT potential fields for inspection, never a motion-safety certificate.

Adapted from pinned CAT pf.py (Apache-2.0; see upstream/NOTICE.md).
Unknown cells block guidance. Unreachable cells never acquire fabricated arrows.
"""
from dataclasses import dataclass
import numpy as np
import skfmm
from scipy.ndimage import map_coordinates

from perception_core import SHAPE, RESOLUTION


@dataclass(frozen=True)
class Fields:
    sdf: np.ndarray
    boundary: np.ndarray
    guidance: np.ndarray
    reachable: np.ndarray
    known_free: np.ndarray
    origin: np.ndarray
    goal: object


def compute_fields(state, processed, origin, goal=None):
    state, processed = np.asarray(state), np.asarray(processed)
    origin = np.asarray(origin, dtype=float)
    if state.shape != SHAPE or processed.shape != SHAPE or origin.shape != (3,):
        raise ValueError("invalid field geometry")
    if not np.isin(state, [0, 1, 2]).all() or not np.isin(processed, [0, 1]).all():
        raise ValueError("invalid occupancy values")
    if not np.isfinite(origin).all():
        raise ValueError("invalid grid origin")
    # Retain observed thin obstacles removed by morphology; never interpret
    # unseen space as free. This intentional difference is reported by the node.
    blocked = (state != 1) | processed.astype(bool)
    free = ~blocked
    if not free.any() or blocked.all():
        raise ValueError("no observed free space")
    if not blocked.any():
        # No zero crossing: no invented corner obstacles to satisfy skfmm.
        sdf = np.full(SHAPE, np.linalg.norm(np.array(SHAPE)*RESOLUTION), np.float32)
        boundary = np.zeros(SHAPE+(3,), np.float32)
    else:
        phi = np.where(blocked, -1., 1.)
        sdf = skfmm.distance(phi, dx=RESOLUTION).astype(np.float32)
        boundary = np.stack(np.gradient(sdf, RESOLUTION, edge_order=2), axis=-1).astype(np.float32)
    guidance = np.zeros(SHAPE+(3,), np.float32)
    reachable = np.zeros(SHAPE, bool)
    if goal is not None:
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("invalid goal")
        index = np.floor((goal-origin)/RESOLUTION).astype(int)
        if (index < 0).any() or (index >= SHAPE).any():
            raise ValueError("goal is outside the local volume")
        if blocked[tuple(index)]:
            raise ValueError("goal is occupied or unknown")
        axes = [origin[i]+(np.arange(SHAPE[i])+.5)*RESOLUTION for i in range(3)]
        xyz = np.meshgrid(*axes, indexing='ij')
        seed = sum((xyz[i]-goal[i])**2 for i in range(3)) <= .12**2
        seed &= free
        if not seed.any() or not (free & ~seed).any():
            raise ValueError("goal seed has no traversable neighborhood")
        phi = np.ma.MaskedArray(np.where(seed, -1., 1.), mask=blocked)
        arrival = skfmm.distance(phi, dx=RESOLUTION)
        reachable = free & ~np.ma.getmaskarray(arrival)
        values = np.ma.filled(arrival, float(np.ma.max(arrival))).astype(np.float32)
        g = -np.stack(np.gradient(values, RESOLUTION, edge_order=2), axis=-1)
        b = boundary/(np.linalg.norm(boundary, axis=-1, keepdims=True)+1e-9)
        tangent = g - np.sum(g*b, axis=-1, keepdims=True)*b
        t = np.clip(sdf/(5*RESOLUTION+1e-9), 0, 1)
        weight = (1-t*t*(3-2*t))[..., None]
        mixed = (1-weight)*g + weight*tangent
        norm = np.linalg.norm(mixed, axis=-1, keepdims=True)
        direction = np.divide(mixed, norm, out=np.zeros_like(mixed), where=norm > 1e-9)
        xy_distance = np.sqrt((xyz[0]-goal[0])**2 + (xyz[1]-goal[1])**2)
        # Upstream field magnitude, NOT an issued speed command or safety limit.
        speed = .6*np.minimum(1., (xy_distance/.3)**3)
        guidance = (direction*speed[..., None]).astype(np.float32)
        guidance[~reachable] = 0
    for array in (sdf, boundary, guidance):
        if not np.isfinite(array).all():
            raise ValueError("nonfinite potential field")
        array.flags.writeable = False
    return Fields(sdf, boundary, guidance, reachable, free, origin.copy(), goal)


def sample_fields(fields, positions):
    """Trilinear voxel-center sampling; fail closed outside observed support."""
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("invalid body sample locations")
    idx = (points-fields.origin)/RESOLUTION-.5
    if (idx < 0).any() or (idx > np.array(SHAPE)-1).any():
        raise ValueError("body sample outside grid-center support")
    # Require every interpolation corner known free, not just the nearest voxel.
    low, high = np.floor(idx).astype(int), np.ceil(idx).astype(int)
    for bits in np.ndindex(2, 2, 2):
        corner = np.where(np.array(bits), high, low)
        if not fields.known_free[tuple(corner.T)].all():
            raise ValueError("body sample intersects occupied or unknown space")
    def sample(a):
        return map_coordinates(a, idx.T, order=1, mode='nearest', prefilter=False)
    return dict(sdf=sample(fields.sdf),
                boundary=np.column_stack([sample(fields.boundary[..., i]) for i in range(3)]),
                guidance=np.column_stack([sample(fields.guidance[..., i]) for i in range(3)]))
