"""Occupancy -> policy fields on the robot, with the same window/goal rules as cat/wholebody_sim.py.

VirtualScene places a training scene (box list, cat/wholebody/scenes/*.json) in the map frame: the robot walks
and localises in the real room, but the policy sees only these boxes (esdf_map_virtual). FieldPlanner reproduces
wholebody_sim.update_fields (rolling 128x128x38 window at 4 cm, waypoint goal at z = 0.75 m projected into the
window, nearest free cell if the goal is blocked) and the waypoint advance rule. FieldWorker computes fields in a
separate process and hands them over through shared memory: one computation takes ~0.8 s on the laptop and must
never stall the 50 Hz control loop.
"""
import json
import math
import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from wholebody_core import FieldGrid, WHOLEBODY
from wholebody_fields import goal_gradient, mix_guidance, training_fields
from wholebody_sim import DX, WINDOW, box_corners_test

from .esdf_layers import REAL_Z, StaticEsdf, band, combine, dynamic_voxels, fmm_sdf, influence_box, remove_body, stamp_sdf, union_box

GOAL_Z = .75
OBSTACLE_MODES = ('virtual', 'real', 'both')


def scene_dirs():
    """Extra scenes first (G1_SCENES_DIR, e.g. runtime/wholebody/scenes on the robot: drop a sim JSON there), then the
    training scenes shipped in cat/wholebody/scenes."""
    extra = os.environ.get('G1_SCENES_DIR')
    return ([Path(extra)] if extra else []) + [WHOLEBODY / 'scenes']


def available_scenes():
    names = []
    for d in scene_dirs():
        if d.is_dir():
            names += [f.stem for f in sorted(d.glob('*.json')) if f.stem not in names]
    return names


def load_scene(name):
    path = Path(name)
    if path.suffix == '.json':
        return json.loads(path.read_text())
    for d in scene_dirs():
        if (d / (name + '.json')).is_file():
            return json.loads((d / (name + '.json')).read_text())
    raise FileNotFoundError('scene %s not in %s' % (name, ', '.join(map(str, scene_dirs()))))


def points_occupancy(points, first_center, shape, z_range=REAL_Z):
    """Real obstacle points (map frame, floor at z = 0) -> occupancy of the window: one voxel per point."""
    occ = np.zeros(shape, bool)
    if points is None or len(points) == 0:
        return occ
    p = np.asarray(points, float)[:, :3]
    idx = np.floor((p - (np.asarray(first_center) - DX / 2)) / DX).astype(int)
    ok = (idx >= 0).all(1) & (idx < np.asarray(shape)).all(1) & (p[:, 2] > z_range[0]) & (p[:, 2] < z_range[1])
    occ[tuple(idx[ok].T)] = True
    return occ


class VirtualScene:
    """Scene boxes moved into the map: p_map = R(yaw) p_scene + (x, y, 0); box yaw += anchor yaw.

    anchor = where the scene's start point goes in the map, and which way the scene's +x points. With
    anchor_at_start=True (default) the scene start itself is mapped to (x, y): put (x, y, yaw) at the robot's
    starting pose and the scene is laid out in front of it exactly as in simulation."""

    def __init__(self, scene, anchor=(0., 0., 0.), anchor_at_start=True, robot_perception=False):
        self.spec = load_scene(scene) if isinstance(scene, (str, Path)) else scene
        x, y, yaw = (float(v) for v in anchor)
        c, s = math.cos(yaw), math.sin(yaw)
        self.rotation = np.array([[c, -s], [s, c]])
        start = np.asarray(self.spec['start'][:2], float)
        self.offset = np.array([x, y]) - (self.rotation @ start if anchor_at_start else 0.)
        self.yaw = yaw
        self.boxes = []
        for b in self.spec['boxes']:
            center = np.asarray(b['center'], float).copy()
            center[:2] = self.to_map(center[:2])
            self.boxes.append(dict(b, center=center.tolist(), yaw=float(b.get('yaw', 0.)) + yaw))
        self.start = self.to_map(start)
        self.goal = self.to_map(np.asarray(self.spec['goal'][:2], float))
        self.robot_perception = robot_perception
        self.name = self.spec.get('scene_id', 'scene')
        # what it was asked for by (the file name the viewer's list uses); scene_id may differ (training-bank ids)
        self.key = Path(scene).stem if isinstance(scene, (str, Path)) else self.name

    def to_map(self, xy):
        return self.rotation @ np.asarray(xy, float) + self.offset

    def occupancy(self, first_center, shape):
        axes = [first_center[i] + np.arange(shape[i]) * DX for i in range(3)]
        centers = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        occ = box_corners_test(self.boxes, centers, DX).reshape(shape)
        if self.robot_perception:                       # what the real LiDAR volume would keep (sim option)
            z = axes[2]; occ[:, :, z < .10] = False; occ[:, :, z > 1.4] = False
        return occ

    def waypoints(self, spacing=1.6):
        d = self.goal - self.start; n = max(1, int(math.ceil(np.linalg.norm(d) / spacing)))
        return [self.start + d * (k / n) for k in range(1, n + 1)]

    def inside_any_box(self, points, margin=0.):
        """Which points lie inside a (yawed) box: virtual 'contacts' of robot sites in tests and logs."""
        points = np.atleast_2d(points); hit = np.zeros(len(points), bool)
        for b in self.boxes:
            c, h, yaw = np.asarray(b['center']), np.asarray(b['half_size']) + margin, b['yaw']
            d = points - c; ca, sa = math.cos(yaw), math.sin(yaw)
            lx, ly = d[:, 0] * ca + d[:, 1] * sa, -d[:, 0] * sa + d[:, 1] * ca
            hit |= (np.abs(lx) < h[0]) & (np.abs(ly) < h[1]) & (np.abs(d[:, 2]) < h[2])
        return hit


def window_request(base_xy, goal_xy, occupancy):
    """wholebody_sim.update_fields minus the field solve: window origin, occupancy, projected goal."""
    center = np.round(np.asarray(base_xy, float) / DX) * DX
    first = np.array([center[0] - WINDOW[0] / 2 * DX + DX / 2, center[1] - WINDOW[1] / 2 * DX + DX / 2, DX / 2])
    goal = np.r_[np.asarray(goal_xy, float), GOAL_Z]
    lo, hi = first[:2] + .3, first[:2] + (np.array(WINDOW[:2]) - 1) * DX - .3
    goal[:2] = np.clip(goal[:2], lo, hi)
    occ = occupancy(first, WINDOW)
    free = ~occ[..., int(round((GOAL_Z - DX / 2) / DX))]
    cell = tuple(np.clip(np.round((goal[:2] - first[:2]) / DX).astype(int), 0, np.array(WINDOW[:2]) - 1))
    if not free[cell]:
        idx = np.argwhere(free); target = np.round((goal[:2] - first[:2]) / DX)
        goal[:2] = first[:2] + idx[np.argmin(((idx - target) ** 2).sum(-1))] * DX
    return first, occ, goal


def solve_fields(base_xy, goal_xy, occupancy, goal_seed='ball'):
    first, occ, goal = window_request(base_xy, goal_xy, occupancy)
    return training_fields(occ, first, DX, goal, goal_seed=goal_seed), goal


class FieldPlanner:
    """Waypoints + when to recompute (sim rules): every `period` s and whenever the active waypoint advances."""

    def __init__(self, scene, spacing=1.6, advance_radius=.8, reach_radius=.4, period=1.):
        self.scene, self.spacing = scene, spacing
        self.waypoints = scene.waypoints(spacing)
        self.operator_goal = self.route_start = None      # a goal set from the viewer instead of the scene's
        self.active = 0; self.advance_radius, self.reach_radius, self.period = advance_radius, reach_radius, period
        self.reached = False
        self.hold_xy = None            # operator 'stop': goal = where the robot is

    def set_route(self, start_xy, goal_xy):
        """Operator goal: a straight route from start_xy, waypoints every `spacing` as for a scene."""
        s, g = np.asarray(start_xy, float)[:2], np.asarray(goal_xy, float)[:2]
        d = g - s; n = max(1, int(math.ceil(np.linalg.norm(d) / self.spacing)))
        self.waypoints = [s + d * (k / n) for k in range(1, n + 1)]
        self.active, self.reached, self.operator_goal, self.route_start = 0, False, g, s

    @property
    def goal(self):
        return self.hold_xy if self.hold_xy is not None else self.waypoints[self.active]

    def update(self, base_xy):
        """Returns True when the waypoint advanced (fields must be recomputed now, policy re-armed if stopped)."""
        xy = np.asarray(base_xy, float)[:2]
        advanced = False
        if self.hold_xy is None and self.active < len(self.waypoints) - 1 and np.linalg.norm(self.waypoints[self.active] - xy) < self.advance_radius:
            self.active += 1; advanced = True
        if self.active == len(self.waypoints) - 1 and np.linalg.norm(self.waypoints[-1] - xy) < self.reach_radius:
            self.reached = True
        return advanced


# ------------------------------------------------------------------------------------------------ worker process
_SHAPE = tuple(WINDOW)
_N = int(np.prod(_SHAPE))
_SLOT_BYTES = 44 * _N          # sdf 4 | gf 12 | bf 12 | raw guidance g 12 | base sdf 4 (bytes per voxel)


def compose(mode, scene, static, points, body, first, shape):
    """The window's (base sdf, sdf, blocked, dynamic box). base: what does not change between solves (the static map
    crop and/or the virtual boxes; with no map in 'real', the lidar window of this solve). sdf = min(base, dynamic
    lidar); the box bounds where the dynamic voxels changed it (esdf_layers.influence_box)."""
    if mode == 'virtual':
        base = fmm_sdf(scene.occupancy(first, shape))
        return base, base, base < 0, None
    pts = remove_body(band(points), body)
    occ_v = scene.occupancy(first, shape) if mode == 'both' else None
    if static is not None:
        base = static.crop(first, shape)
        if occ_v is not None and occ_v.any():
            base = np.minimum(base, fmm_sdf(occ_v))
    else:
        occ = points_occupancy(pts, first, shape)
        base = fmm_sdf(occ if occ_v is None else occ | occ_v)
    idx = dynamic_voxels(pts, first, shape, base)
    sdf, blocked = combine(base, idx)
    return base, sdf, blocked, influence_box(idx, shape)


def _apply_roi(out, roi, idx, first, goal):
    lo, hi = roi
    lo2, hi2 = np.maximum(lo - 2, 0), np.minimum(hi + 2, np.asarray(_SHAPE))
    sl2 = tuple(slice(a, b) for a, b in zip(lo2, hi2))
    base = out[4][sl2]
    sub = base if len(idx) == 0 else np.minimum(base, stamp_sdf(idx - lo2, base.shape))
    gf, bf = mix_guidance(sub, out[3][sl2], sub < 0, DX, np.asarray(first, float) + lo2 * DX, goal)
    inner = tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, lo2))
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    out[0][sl] = sub[inner]; out[1][sl] = gf[inner]; out[2][sl] = bf[inner]


def _worker(requests, results, names, scene_spec, scene_kwargs, static_path, goal_seed='ball'):
    scene, mode = VirtualScene(scene_spec, **scene_kwargs), 'virtual'
    static = StaticEsdf(static_path) if static_path else None
    blocks = [shared_memory.SharedMemory(name=n) for n in names]
    views = [_views(b.buf) for b in blocks]
    while True:
        req = requests.get()
        if req is None:
            break
        if req[0] == 'world':                               # same queue as the requests: applied in order
            _, mode, spec, kwargs = req
            if spec is not None:
                scene = VirtualScene(spec, **kwargs)
            continue
        kind, seq, t = req[0], req[1], time.monotonic()
        try:
            if kind == 'field':                             # full solve: window, goal projection, geodesic (slow)
                _, _, slot, base_xy, goal_xy, points, body = req
                cache = {}

                def occupancy(first, shape):
                    cache['r'] = compose(mode, scene, static, points, body, first, shape)
                    return cache['r'][2]
                first, blocked, goal = window_request(base_xy, goal_xy, occupancy)
                base, sdf, _, box = cache['r']
                if blocked.all():
                    raise ValueError('no free space')
                out = views[slot]
                g = goal_gradient(blocked, first, DX, goal, goal_seed)
                gf, bf = mix_guidance(sdf, g, blocked, DX, first, goal)
                out[0][...] = sdf; out[1][...] = gf; out[2][...] = bf; out[3][...] = g; out[4][...] = base
            else:                                           # 'refresh': same window and geodesic, fresh lidar (fast)
                _, _, src, slot, first, goal, src_box, points, body = req
                out = views[slot]
                for a, b in zip(out, views[src]):
                    a[...] = b
                idx = dynamic_voxels(remove_body(band(points), body), first, _SHAPE, out[4])
                box = influence_box(idx, _SHAPE)
                roi = union_box(box, None if src_box is None else (np.asarray(src_box[0]), np.asarray(src_box[1])))
                if roi is not None:
                    _apply_roi(out, roi, idx, first, goal)
            box = None if box is None else (box[0].tolist(), box[1].tolist())
            results.put((kind, seq, slot, list(map(float, first)), DX, list(map(float, goal)), box, time.monotonic() - t, None))
        except Exception as exc:  # reported, never silently stale: the control side ages the old fields out
            results.put((kind, seq, None, None, None, None, None, time.monotonic() - t, repr(exc)))
    for b in blocks:
        b.close()


def _views(buf):
    n = _N
    sdf = np.ndarray(_SHAPE, np.float32, buf, 0)
    gf = np.ndarray(_SHAPE + (3,), np.float32, buf, 4 * n)
    bf = np.ndarray(_SHAPE + (3,), np.float32, buf, 16 * n)
    g = np.ndarray(_SHAPE + (3,), np.float32, buf, 28 * n)
    base = np.ndarray(_SHAPE, np.float32, buf, 40 * n)
    return sdf, gf, bf, g, base


class FieldWorker:
    """Field computation in `n` spawned solver processes plus one refresh process, through shared memory.

    poll() returns the newest full solve (FieldGrid, goal). In 'real' / 'both', refresh(grid) re-makes the grid the
    control loop is using from the newest lidar every REFRESH_PERIOD_S (same window and geodesic, new sdf / boundary /
    guidance blending, ~10-50 ms) and take_refreshed() hands back (that grid, its refresh). No slot that the control
    loop may read (hold()) or that is in flight is ever written. max_pending = how many full solves may run at once
    (1 normally; up to n for the joystick's fast goals). static: a StaticEsdf npz (esdf_layers) for 'real' / 'both'."""

    REFRESH_PERIOD_S = 0.1

    def __init__(self, scene, n=1, static=None, dynamic=True, goal_seed='ball', **scene_kwargs):
        ctx = mp.get_context('spawn')                   # no fork of a process that has DDS threads
        self.goal_seed = goal_seed                      # the policy's training seed (contract "goal_seed": ball | column)
        self.n = max(1, int(n)); self.max_pending = 1
        self.blocks = [shared_memory.SharedMemory(create=True, size=_SLOT_BYTES) for _ in range(self.n + 5)]
        self.views = [_views(b.buf) for b in self.blocks]
        self.queues, self.results = [ctx.Queue() for _ in range(self.n + 1)], ctx.Queue()    # the last: refreshes
        spec = scene.spec if isinstance(scene, VirtualScene) else load_scene(scene)
        if isinstance(scene, VirtualScene):
            scene_kwargs = dict(anchor=(scene.offset[0], scene.offset[1], scene.yaw), anchor_at_start=False,
                                robot_perception=scene.robot_perception)
        self.static_path = str(static) if static else None
        self.dynamic = bool(dynamic)    # False: 'real' / 'both' = the static map only (no lidar layer, no refresh)
        if not self.dynamic and not self.static_path:
            raise ValueError('no lidar layer needs a static map')
        if self.static_path:
            StaticEsdf(self.static_path)                # fail here, not in the workers
        self.processes = [ctx.Process(target=_worker, daemon=True,
                                      args=(q, self.results, [b.name for b in self.blocks], spec, scene_kwargs, self.static_path,
                                            goal_seed))
                          for q in self.queues]
        for pr in self.processes:
            pr.start()
        self.in_use, self.flight, self.seq, self.delivered, self.turn = None, {}, 0, 0, 0
        self.last_compute_s, self.last_error = None, None
        self.mode, self.scene, self.world_seq = 'virtual', scene, 0
        self.points_source = None       # real / both: callable -> map-frame points (N, 3), or (N, 4) with each point's scan
                                        # time (system clock), or None when there are none fresh
        self.body_source = None         # callable -> the robot for the self-filter (esdf_layers.remove_body): capsules, or the
                                        # pose-history dict (t, q, pos, quat, ...); None = no pose yet (no solve)
        self.held, self.refreshing, self.refreshed, self.next_refresh = set(), None, None, 0.
        self.last_refresh_s, self.refreshes, self.refresh_started = None, 0, 0.

    @property
    def pending(self):
        return next(iter(self.flight.values()), None)

    def _free_slot(self, *extra):
        busy = {slot for slot, _ in self.flight.values()} | {self.in_use} | self.held | set(extra)
        if self.refreshing is not None:
            busy |= {self.refreshing[1].slot, self.refreshing[2]}
        if self.refreshed is not None:
            busy |= {self.refreshed[0].slot, self.refreshed[1].slot}
        return next((i for i in range(len(self.blocks)) if i not in busy), None)

    def _inputs(self):
        """(points, body) for 'real' / 'both', or None when there is no fresh cloud (never solve from nothing)."""
        points = self.points_source() if self.points_source is not None else None
        if points is None:
            self.last_error = 'no fresh real obstacle cloud'
            return None
        body = self.body_source() if self.body_source is not None else None
        if self.body_source is not None and body is None:     # no pose yet: the robot and the rope would be obstacles
            self.last_error = 'no body pose for the lidar self-filter yet'
            return None
        return np.asarray(points, float), body          # float64: column 3 may be an absolute time

    def set_world(self, mode=None, scene=None):
        """Switch the obstacle source and/or the virtual scene. Answers to requests made before are dropped."""
        mode = self.mode if mode is None else mode
        if mode not in OBSTACLE_MODES:
            raise ValueError('obstacle mode %s' % mode)
        spec = kwargs = None
        if scene is not None:
            spec = scene.spec
            kwargs = dict(anchor=(scene.offset[0], scene.offset[1], scene.yaw), anchor_at_start=False,
                          robot_perception=scene.robot_perception)
            self.scene = scene
        self.mode = mode; self.world_seq = self.seq + 1
        for q in self.queues:
            q.put(('world', mode, spec, kwargs))

    def request(self, base_xy, goal_xy):
        if len(self.flight) >= min(self.max_pending, self.n):
            return False
        points = body = None
        if self.mode != 'virtual' and self.dynamic:
            inputs = self._inputs()
            if inputs is None:
                return False
            points, body = inputs
        slot = self._free_slot()
        if slot is None:
            return False
        self.seq += 1; self.flight[self.seq] = (slot, time.monotonic())
        self.queues[self.turn].put(('field', self.seq, slot, list(map(float, base_xy[:2])), list(map(float, goal_xy[:2])), points, body))
        self.turn = (self.turn + 1) % self.n
        return True

    def hold(self, grids):
        """The grids the control loop may read: their slots are never written."""
        self.held = {g.slot for g in grids if g is not None and getattr(g, 'slot', None) is not None}

    def refresh(self, grid, now):
        """Start re-making `grid` from the newest lidar (real / both), at most every REFRESH_PERIOD_S."""
        if self.refreshing is not None and time.monotonic() - self.refresh_started > 2.:
            # no answer: reported, not retried (its slot may still be written). The control side sees the obstacles go stale.
            self.last_error = 'lidar refresh %s' % ('process died' if not self.processes[-1].is_alive() else 'not answering')
        if self.mode == 'virtual' or not self.dynamic or grid is None or getattr(grid, 'slot', None) is None or self.refreshing is not None \
                or self.refreshed is not None or now < self.next_refresh:
            return False
        inputs = self._inputs()
        if inputs is None:
            return False
        slot = self._free_slot(grid.slot)
        if slot is None:
            return False
        self.seq += 1; self.refreshing = (self.seq, grid, slot); self.next_refresh = now + self.REFRESH_PERIOD_S
        self.refresh_started = time.monotonic()
        self.queues[-1].put(('refresh', self.seq, grid.slot, slot, list(map(float, grid.origin)), list(map(float, grid.goal)),
                             grid.box, *inputs))
        return True

    def take_refreshed(self):
        out, self.refreshed = self.refreshed, None
        return out

    def _grid(self, slot, origin, dx, goal, box):
        sdf, gf, bf, _, _ = self.views[slot]
        grid = FieldGrid(sdf, gf, bf, origin, dx); grid.slot = slot; grid.goal = np.asarray(goal); grid.box = box
        return grid

    def poll(self):
        new = None
        while not self.results.empty():
            kind, seq, slot, origin, dx, goal, box, seconds, error = self.results.get_nowait()
            if kind == 'refresh':
                src, self.refreshing = self.refreshing, None
                self.last_refresh_s = seconds
                if error is not None:
                    self.last_error = error
                elif src is not None and src[0] == seq and seq >= self.world_seq:
                    self.refreshed = (src[1], self._grid(slot, origin, dx, goal, box)); self.refreshes += 1
                continue
            self.flight.pop(seq, None); self.last_compute_s = seconds; self.last_error = error
            if seq < self.world_seq or seq < self.delivered:   # another world, or overtaken by a newer answer
                continue
            if error is None:
                self.in_use, self.delivered = slot, seq
                new = (self._grid(slot, origin, dx, goal, box), np.asarray(goal))
        return new

    def close(self):
        try:
            for q in self.queues:
                q.put(None)
            for pr in self.processes:
                pr.join(timeout=5)
        finally:
            for pr in self.processes:
                if pr.is_alive():
                    pr.terminate()
            for b in self.blocks:
                b.close(); b.unlink()
