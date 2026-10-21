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
from wholebody_fields import training_fields
from wholebody_sim import DX, WINDOW, box_corners_test

GOAL_Z = .75
OBSTACLE_MODES = ('virtual', 'real', 'both')
REAL_Z = (0.10, 1.40)          # real points kept as obstacles: above the floor, below the window top (as sim's robot_perception)


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
    p = np.asarray(points, float)
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


def solve_fields(base_xy, goal_xy, occupancy):
    first, occ, goal = window_request(base_xy, goal_xy, occupancy)
    return training_fields(occ, first, DX, goal), goal


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


def _worker(requests, results, names, scene_spec, scene_kwargs):
    scene, mode = VirtualScene(scene_spec, **scene_kwargs), 'virtual'
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
        _, seq, slot, base_xy, goal_xy, points = req
        t = time.monotonic()
        if mode == 'virtual':
            occupancy = scene.occupancy
        elif mode == 'real':
            occupancy = lambda first, shape: points_occupancy(points, first, shape)
        else:
            occupancy = lambda first, shape: scene.occupancy(first, shape) | points_occupancy(points, first, shape)
        try:
            fields, goal = solve_fields(base_xy, goal_xy, occupancy)
            sdf, gf, bf = views[slot]
            sdf[...] = fields.sdf; gf[...] = fields.gf; bf[...] = fields.bf
            results.put((seq, slot, fields.origin.tolist(), fields.dx, goal.tolist(), time.monotonic() - t, None))
        except Exception as exc:  # reported, never silently stale: the control side ages the old fields out
            results.put((seq, slot, None, None, None, time.monotonic() - t, repr(exc)))
    for b in blocks:
        b.close()


def _views(buf):
    n = int(np.prod(_SHAPE))
    sdf = np.ndarray(_SHAPE, np.float32, buf, 0)
    gf = np.ndarray(_SHAPE + (3,), np.float32, buf, 4 * n)
    bf = np.ndarray(_SHAPE + (3,), np.float32, buf, 16 * n)
    return sdf, gf, bf


class FieldWorker:
    """Double-buffered field computation in a spawned process. poll() returns a new FieldGrid when one is ready.
    The buffer the control loop is using is never written: requests always target the other slot."""

    def __init__(self, scene, **scene_kwargs):
        ctx = mp.get_context('spawn')                   # no fork of a process that has DDS threads
        size = 28 * int(np.prod(_SHAPE))
        self.blocks = [shared_memory.SharedMemory(create=True, size=size) for _ in range(2)]
        self.views = [_views(b.buf) for b in self.blocks]
        self.requests, self.results = ctx.Queue(), ctx.Queue()
        spec = scene.spec if isinstance(scene, VirtualScene) else load_scene(scene)
        if isinstance(scene, VirtualScene):
            scene_kwargs = dict(anchor=(scene.offset[0], scene.offset[1], scene.yaw), anchor_at_start=False,
                                robot_perception=scene.robot_perception)
        self.process = ctx.Process(target=_worker, args=(self.requests, self.results, [b.name for b in self.blocks],
                                                         spec, scene_kwargs), daemon=True)
        self.process.start()
        self.in_use, self.pending, self.seq = None, None, 0
        self.last_compute_s, self.last_error = None, None
        self.mode, self.scene, self.world_seq = 'virtual', scene, 0
        self.points_source = None       # real / both: callable -> (N, 3) map-frame points, or None when there are none fresh

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
        self.requests.put(('world', mode, spec, kwargs))

    def request(self, base_xy, goal_xy):
        if self.pending is not None:
            return False
        points = None
        if self.mode != 'virtual':
            points = self.points_source() if self.points_source is not None else None
            if points is None:                  # never solve 'real' from nothing: it would read as free space
                self.last_error = 'no fresh real obstacle cloud'
                return False
            points = np.asarray(points, np.float32)
        slot = 0 if self.in_use != 0 else 1
        self.seq += 1; self.pending = (self.seq, time.monotonic())
        self.requests.put(('field', self.seq, slot, list(map(float, base_xy[:2])), list(map(float, goal_xy[:2])), points))
        return True

    def poll(self):
        new = None
        while not self.results.empty():
            seq, slot, origin, dx, goal, seconds, error = self.results.get_nowait()
            self.pending = None; self.last_compute_s = seconds; self.last_error = error
            if seq < self.world_seq:            # solved for the previous scene / obstacle source
                continue
            if error is None:
                sdf, gf, bf = self.views[slot]
                self.in_use = slot
                new = (FieldGrid(sdf, gf, bf, origin, dx), np.asarray(goal))
        return new

    def close(self):
        try:
            self.requests.put(None); self.process.join(timeout=5)
        finally:
            if self.process.is_alive():
                self.process.terminate()
            for b in self.blocks:
                b.close(); b.unlink()
