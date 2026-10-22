#!/usr/bin/env python3
"""Replay of a demo take from the outside: the G1 model (MuJoCo, training robot.xml + Unitree meshes) posed from the
recorded data, in the virtual scene and the room, seen from the side and from above.

  MUJOCO_GL=egl python3 replay_take.py <take_dir> --meshes <g1_description/meshes> [--fps 30] [--out <take>/replay.mp4]
                                       [--full] [--slow 0.4] [--stills 2,8,12]

Pose (200 Hz track, precomputed):
  joints     the 29 joints from the 1 kHz motor log (<take>/raw_motor_1khz, LowState q on the drive's 1 ms tick);
  roll/pitch the pelvis IMU (1 kHz);
  heading    the IMU yaw plus its offset to the SLAM heading of /g1_wb/pelvis (offset smoothed over ~1 s, so the
             heading is the SLAM one without the 5 Hz steps);
  x, y       leg odometry from the stance foot, pulled to the SLAM pelvis with a 0.6 s time constant (the 5 Hz SLAM
             pose alone makes the planted foot skate by ~0.2 m/s; this keeps it planted within ~0.2 m of SLAM);
  z          the lowest corner of the two foot boxes on the floor.
Scene: the bag's /g1_wb/virtual_scene boxes (orange), the room from /g1_slam/map as a floor plan of occupied 10 cm
cells (grey, lighter and thinned when the room is cluttered), the planned path and the goal (green), the pelvis trail
(blue). The presentation follows the obstacles:
  beam  (a box with its bottom above the floor): left, an orthographic side view along the beam's axis (true
        heights: the beam bottom against the robot, the clearance under it); right, a view from above from the same
        side (nearly along the beam, so it never stands between camera and robot);
  walls (boxes standing on the floor): left, an orthographic top view, walk left to right, with the gap width
        and the live clearance of the closest body point (mesh vertices) to each wall; right, a view from above
        from behind, into the gap.
The AR head camera is inset on the right. By default the clip runs from shortly before the robot starts walking
to shortly after it stops (later moves, such as a collapse on DAMP, are left out), with the crouch or the passage
in slow motion (--full: whole take, --slow 1: real time). --stills writes PNGs of the given take times instead.
If the bag missed /tf_static, map -> camera_init (for the room) is estimated from the standing moments of the take.
Needs numpy, mujoco, Pillow and ffmpeg (no ROS, no OpenCV).
"""
import argparse
import glob
import io
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bagread import Cdr, latest, load, read_cloud, read_compressed, read_markers, read_pose_stamped, read_string, read_tf, rot, yaw

HERE = Path(__file__).resolve().parent
ROBOT_XML = HERE.parents[1] / 'cat' / 'wholebody' / 'robot.xml'
PW, PH, TOP = 960, 1004, 76               # panel size, title band -> 1920 x 1080
DT = 0.005                                # pose track step
SIDE_EXTENT = 2.05                        # side view: visible height (m), orthographic
SIDE_ZC = 0.75                            # side view: height of the image centre (m)
FLOOR_RGB = (196, 200, 206)
INK, PAPER = (255, 255, 255), (24, 30, 40)
ORANGE, BLUE, GREEN = (240, 120, 30), (40, 115, 230), (40, 190, 80)
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')


def font(size, bold=False):
    try:
        return ImageFont.truetype(str(FONT_DIR / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), size)
    except OSError:
        return ImageFont.load_default()


def quat_wxyz(r):
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1) * 2; return np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    i = int(np.argmax(np.diag(r)))
    if i == 0:
        s = math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]) * 2; return np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2]) * 2; return np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
    s = math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1]) * 2; return np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def gsmooth(x, sigma, dt=DT):
    """Gaussian smoothing (non-causal, edges held) of a 1-D or (n, k) series sampled every dt."""
    k = np.arange(-int(4 * sigma / dt), int(4 * sigma / dt) + 1) * dt; g = np.exp(-0.5 * (k / sigma) ** 2); g /= g.sum()
    x = np.asarray(x, float); one = x.ndim == 1; x = x.reshape(len(x), -1); n = len(k) // 2
    out = np.column_stack([np.convolve(np.pad(c, (n, n), mode='edge'), g, mode='valid') for c in x.T])
    return out[:, 0] if one else out


def motor_log(take):
    """LowState q and pelvis IMU from the 1 kHz motor log on the robot's wall clock (tick = 1 ms; duplicates dropped)."""
    d = {}
    for f in sorted(glob.glob(str(Path(take) / 'raw_motor_1khz' / 'motors_*.npz'))):
        z = np.load(f)
        for k in ('t', 'tick', 'q', 'quat'):
            d.setdefault(k, []).append(z['state_' + k])
    d = {k: np.concatenate(v) for k, v in d.items()}
    tick = d['tick'].astype(np.int64); keep = np.r_[True, np.diff(tick) != 0]
    d = {k: v[keep] for k, v in d.items()}; tick = tick[keep]
    wall = np.min(d['t'] - tick / 1000.) + tick / 1000.        # batch receive stamps -> the earliest is the closest
    return wall, d['q'], d['quat']


# lidar body -> pelvis while standing (calibrated on take 20261021_182717 against its map -> camera_init); used to
# recover map -> camera_init when a bag missed the latched /tf_static
X_BODY_PELVIS = ((0.0205, -0.0237, 0.515), (0.9868, -0.0156, -0.1613, -0.0019))


def hom(t, q):
    out = np.eye(4); out[:3, :3] = rot(q); out[:3, 3] = t
    return out


def mean_pose(ts):
    u, _, vt = np.linalg.svd(sum(t[:3, :3] for t in ts)); out = np.eye(4); out[:3, :3] = u @ vt
    out[:3, 3] = np.mean([t[:3, 3] for t in ts], 0)
    return out


def map_from_camera_init(m):
    """map -> camera_init (the SLAM map's frame) from /tf_static, or estimated from the standing moments of the take:
    pelvis (map) * inv(body -> pelvis) * inv(odometry)."""
    for _, raw in m.get('/tf_static', []):
        for parent, child, t, q in read_tf(raw):
            if parent == 'map' and child == 'camera_init':
                return hom(t, q), 'tf_static'
    def stamped(raw, odom):
        r = Cdr(raw); st = r.header()[0]
        if odom:
            r.string()
        return st, hom(np.array(r.f64(3)), np.array(r.f64(4)))
    od = [stamped(raw, True) for _, raw in m.get('/g1_slam/odometry', [])]; pe = [stamped(raw, False) for _, raw in m['/g1_wb/pelvis']]
    if not od:
        return np.eye(4), 'none'
    ost = np.array([o[0] for o in od]); pp = np.array([p[1][:3, 3] for p in pe]); xi = np.linalg.inv(hom(*X_BODY_PELVIS)); est = []
    for k in range(2, len(pe) - 2):
        if np.linalg.norm(pp[k + 2] - pp[k - 2]) < 0.01:      # standing still
            j = int(np.argmin(abs(ost - pe[k][0]))); est.append(pe[k][1] @ xi @ np.linalg.inv(od[j][1]))
    return (mean_pose(est), 'estimated') if est else (np.eye(4), 'none')


def box_dist(pts, b):
    """Signed distance of points to a box marker (negative inside) and the closest points on the box."""
    r = rot(b['q']); half = b['scale'] / 2; loc = (pts - b['p']) @ r; q = np.abs(loc) - half
    d = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    return d, b['p'] + np.clip(loc, -half, half) @ r.T


def footprint(b, n=60):
    """Points along the outline of a box's footprint (xy)."""
    r = rot(b['q'])[:2, :2]; hx, hy = b['scale'][:2] / 2; s = np.linspace(-1, 1, n)
    loc = np.r_[np.c_[s * hx, -hy + 0 * s], np.c_[s * hx, hy + 0 * s], np.c_[-hx + 0 * s, s * hy], np.c_[hx + 0 * s, s * hy]]
    return b['p'][:2] + loc @ r.T


def wall_gap(walls, track):
    """The passage between two standing boxes nearest to the robot's track: (i, j, A, B, width), A and B the
    closest footprint points of boxes i and j."""
    best = None
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            fa, fb = footprint(walls[i]), footprint(walls[j]); d = np.linalg.norm(fa[:, None] - fb[None], axis=2)
            ia, ib = np.unravel_index(np.argmin(d), d.shape); mid = (fa[ia] + fb[ib]) / 2
            score = np.linalg.norm(track - mid, axis=1).min()
            if best is None or score < best[0]:
                best = (score, i, j, fa[ia], fb[ib], d[ia, ib])
    return best[1:] if best else None


def scene_xml(meshes, boxes, room, tile_rgba='0.66 0.68 0.72 1'):
    xml = ROBOT_XML.read_text().replace('file="assets/', 'file="%s/' % Path(meshes).resolve())
    geoms = ['<geom type="box" pos="%s" size="%s" quat="%s" contype="0" conaffinity="0" material="virtual_box"/>' % (
        ' '.join('%.4f' % v for v in b['p']), ' '.join('%.4f' % (v / 2) for v in b['scale']), ' '.join('%.5f' % v for v in (b['q'][3], *b['q'][:3])))
        for b in boxes]
    # the room as a floor plan: occupied 10 cm cells as flat tiles (real things there -- crane, people, furniture --
    # would hide the robot as boxes)
    geoms += ['<geom type="box" pos="%.3f %.3f 0.003" size="0.05 0.05 0.003" contype="0" conaffinity="0" rgba="%s"/>' % (*p[:2], tile_rgba) for p in room]
    geoms.append('<light directional="true" castshadow="false" pos="0 0 6" dir="0.3 0.4 -1" diffuse="0.35 0.35 0.35" specular="0 0 0"/>')
    xml = re.sub(r'(<worldbody>)', r'\1' + ''.join(geoms), xml, count=1)
    xml = xml.replace('</asset>', '<material name="virtual_box" rgba="1 0.45 0.08 0.62" specular="0.05" shininess="0"/></asset>', 1)
    if '<global ' in xml:                                       # offscreen frame size on the model's own <global>
        return xml.replace('<global ', '<global offwidth="%d" offheight="%d" ' % (PW, PH), 1)
    return xml.replace('</mujoco>', '<visual><global offwidth="%d" offheight="%d"/></visual></mujoco>' % (PW, PH))


def pose_track(model, data, wall, mq, mquat, pt, pxy, pyaw, t0, t1, tau=0.6):
    """Pelvis pose on a DT grid (see the module docstring); also the foot sites relative to the pelvis."""
    import mujoco
    ts = np.arange(t0, t1, DT); idx = np.minimum(np.searchsorted(wall, ts), len(wall) - 1)
    r_imu = np.array([rot((x, y, z, w)) for w, x, y, z in mquat[idx]])
    iyaw = np.unwrap(np.arctan2(r_imu[:, 1, 0], r_imu[:, 0, 0]))
    head = iyaw + gsmooth(np.interp(ts, pt, pyaw) - iyaw, 1.0)
    feet = [model.site(n).id for n in ('left_foot', 'right_foot')]
    soles = [model.geom(n).id for n in ('left_foot', 'right_foot')]
    hs = model.geom_size[soles[0]]
    corners = np.array([[sx * hs[0], sy * hs[1], -hs[2]] for sx in (-1, 1) for sy in (-1, 1)])
    quat = np.zeros((len(ts), 4)); foot = np.zeros((len(ts), 2, 3)); low = np.zeros(len(ts))
    for k in range(len(ts)):
        r = rz(head[k]) @ rz(-iyaw[k]) @ r_imu[k]                 # heading from SLAM (via the IMU yaw), tilt from the IMU
        quat[k] = quat_wxyz(r)
        data.qpos[:3] = 0; data.qpos[3:7] = quat[k]; data.qpos[7:] = mq[idx[k]]
        mujoco.mj_kinematics(model, data)
        foot[k] = data.site_xpos[feet]
        low[k] = min((data.geom_xpos[g] + corners @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min() for g in soles)
    slam = np.column_stack([np.interp(ts, pt, pxy[:, 0]), np.interp(ts, pt, pxy[:, 1])])
    stance = np.argmin(foot[:, :, 2], 1); xy = slam.copy()
    for k in range(1, len(ts)):                                 # leg odometry, pulled to SLAM
        s = stance[k - 1]
        xy[k] = xy[k - 1] - (foot[k, s, :2] - foot[k - 1, s, :2]) + DT / tau * (slam[k - 1] - xy[k - 1])
    return dict(t=ts, quat=quat, xy=xy, z=-low, slam=slam)


def timeline(tr, fps, full, slow, event):
    """Take times of the output frames, and the slow-motion weight per frame (1 = slowest). event: mask on the track
    of the moment to slow down (the crouch, the passage)."""
    t = tr['t']; v = np.linalg.norm(np.gradient(gsmooth(tr['xy'], 0.3), DT, axis=0), axis=1)
    mv = np.r_[False, v > 0.15, False]; seg = np.flatnonzero(np.diff(mv.astype(int))).reshape(-1, 2)   # moving segments
    if full or not len(seg):
        a, b = t[0], t[-1]
    else:                                                       # the walk: from the longest segment on
        keep = np.r_[True, t[seg[1:, 0]] - t[seg[:-1, 1] - 1] > 3.0]   # merge segments less than 3 s apart (pauses)
        seg = np.column_stack([seg[keep, 0], seg[np.r_[keep[1:], True], 1]])
        s0, s1 = seg[np.argmax(seg[:, 1] - seg[:, 0])]          # (later moves, e.g. a collapse on DAMP, are left out)
        a, b = max(t[0], t[s0] - 1.5), min(t[-1], t[s1 - 1] + 2.5)
    low = np.flatnonzero(event & (t >= a) & (t <= b))
    c0, c1 = (t[low[0]] - 0.6, t[low[-1]] + 0.6) if len(low) and slow < 1 else (np.inf, np.inf)
    def weight(x):
        ramp = 0.4
        return float(np.clip(min(x - (c0 - ramp), (c1 + ramp) - x) / ramp, 0, 1))
    out, w = [], []
    x = a
    while x < b:
        out.append(x); w.append(weight(x)); x += (1 - (1 - slow) * w[-1]) / fps
    return np.array(out), np.array(w)


def plate(dr, xy, lines, fonts, pad=10, gap=4, alpha=185):
    """Text lines on a dark rounded plate; returns the plate box."""
    x, y = xy; sizes = [dr.textbbox((0, 0), s, font=f) for s, f in zip(lines, fonts)]
    w = max(b[2] for b in sizes) + 2 * pad; h = sum(b[3] for b in sizes) + gap * (len(lines) - 1) + 2 * pad
    dr.rounded_rectangle((x, y, x + w, y + h), 8, fill=PAPER + (alpha,))
    yy = y + pad
    for s, f, b in zip(lines, fonts, sizes):
        dr.text((x + pad, yy), s, font=f, fill=INK + (255,)); yy += b[3] + gap
    return x, y, x + w, y + h


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('take'); ap.add_argument('--meshes', required=True)
    ap.add_argument('--fps', type=float, default=30.); ap.add_argument('--out')
    ap.add_argument('--full', action='store_true', help='the whole take (default: the walk, +-2 s)')
    ap.add_argument('--slow', type=float, default=0.4, help='playback speed through the crouch / the gap (1: real time)')
    ap.add_argument('--stills', help='comma-separated take times (s): write PNGs next to --out instead of the video')
    a = ap.parse_args(); take = Path(a.take); out = Path(a.out) if a.out else take / 'replay.mp4'
    import mujoco
    m = load(take)
    t_map_ci, how = map_from_camera_init(m)
    if how != 'tf_static':
        print('map -> camera_init: %s (no /tf_static in the bag)' % how)
    pel = [(ts, *read_pose_stamped(raw)) for ts, raw in m['/g1_wb/pelvis']]
    pt = np.array([p[0] for p in pel]); pxy = np.array([p[1][:2] for p in pel]); pyaw = np.unwrap([yaw(p[2]) for p in pel])
    markers = read_markers(m['/g1_wb/virtual_scene'][-1][1])
    boxes = [k for k in markers if k['ns'] == 'virtual_scene' and k['type'] == 1]
    path = next((k['points'] for k in markers if k['ns'] == 'path' and len(k['points']) > 1), np.zeros((0, 3)))
    goal = next((k['points'][-1] for k in markers if k['ns'] == 'waypoints' and len(k['points'])), None)
    scene = ''
    for _, raw in m.get('/g1_wb/status', [])[:20]:
        try:
            scene = json.loads(read_string(raw)).get('scene') or scene
        except ValueError:
            pass
    # the presentation follows the obstacles: overhead boxes (a beam) -> side view against the beam bottom;
    # standing boxes (walls) -> view along the walk through the gap between them
    overhead = [b for b in boxes if b['p'][2] - b['scale'][2] / 2 > 0.3]
    standing = [b for b in boxes if b['p'][2] - b['scale'][2] / 2 <= 0.3]
    gap = wall_gap(standing, pxy) if not overhead and len(standing) >= 2 else None
    kind = 'beam' if overhead else 'walls' if gap else 'plain'

    room, _ = read_cloud(m['/g1_slam/map'][-1][1]); room = room @ t_map_ci[:3, :3].T + t_map_ci[:3, 3]
    room = room[(room[:, 2] > 0.15) & (room[:, 2] < 1.9)]
    _, first, count = np.unique(np.floor(room[:, :2] / 0.10).astype(np.int64), axis=0, return_index=True, return_counts=True)
    room = room[first]
    centre = pxy.mean(0); near_c = np.linalg.norm(room[:, :2] - centre, axis=1) < 4.5
    # clutter: occupied cells per m2 within 1.5 m of the walk (the beam take's lab: ~39, a crowded one: ~57)
    grid = np.stack(np.meshgrid(*[np.arange(c - 6, c + 6, 0.1) for c in centre]), -1).reshape(-1, 2)
    walkd = lambda q: np.min(np.linalg.norm(q[:, None] - pxy[None, ::5], axis=2), axis=1)
    density = (walkd(room[:, :2]) < 1.5).sum() / max((walkd(grid) < 1.5).sum() * 0.01, 1e-6); tile = '0.66 0.68 0.72 1'
    if density > 48:                                            # a cluttered room: the well-seen cells, lighter
        near_c &= count >= np.median(count[near_c]); tile = '0.76 0.78 0.81 1'
        print('room: %.0f cells/m2 -> %d cells kept, lighter' % (density, near_c.sum()))
    room = room[near_c][:6000]
    wall, mq, mquat = motor_log(take)
    frames = m.get('/g1_wb/ar/compressed', [])
    model = mujoco.MjModel.from_xml_string(scene_xml(a.meshes, boxes, room, tile)); data = mujoco.MjData(model)
    rend = mujoco.Renderer(model, PH, PW, max_geom=20000)
    t0, t1 = max(pt[0], wall[0]), min(pt[-1], wall[-1])
    tr = pose_track(model, data, wall, mq, mquat, pt, pxy, pyaw, t0, t1)
    look = gsmooth(tr['xy'], 0.8)                               # calm cameras: smoothed, centred (no lag)
    d = pxy[-1] - pxy[0]; walk = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 0.3 else pyaw[0]
    side = walk + math.pi / 2                                   # side view: across the walk, snapped to the beam's axis
    if boxes:
        by = yaw(boxes[0]['q']); side = by + round((side - by) / (math.pi / 2)) * math.pi / 2
    beam_lo = min((b['p'][2] - b['scale'][2] / 2 for b in boxes), default=None) if kind == 'beam' else None
    event = tr['z'] < 0.68                                      # the crouch
    if kind == 'walls':
        wi, wj, ga, gb, gw = gap; gc = (ga + gb) / 2; ax_ = (gb - ga) / gw; nrm = np.array([-ax_[1], ax_[0]])
        dwalk = (path[-1, :2] - path[0, :2]) if len(path) > 1 else d
        nrm *= 1 if nrm @ dwalk >= 0 else -1
        side = math.atan2(nrm[1], nrm[0])                       # the view looks along the walk, through the gap
        event = np.abs((tr['xy'] - gc) @ nrm) < 0.45            # the passage
        pair = [standing[wi], standing[wj]]
        right = np.array([math.sin(side), -math.cos(side)])
        names = ['left wall', 'right wall'] if (pair[0]['p'][:2] - gc) @ right < 0 else ['right wall', 'left wall']
        print('walls: gap %.3f m at (%.2f, %.2f), view azimuth %.1f deg' % (gw, gc[0], gc[1], math.degrees(side)))
    robot_geoms = [(g, model.mesh_vert[model.mesh_vertadr[model.geom_dataid[g]]:][:model.mesh_vertnum[model.geom_dataid[g]]])
                   for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    fovy = model.vis.global_.fovy

    def pose(t):
        """Pose the model at take time t; the track index and the body's mesh vertices (world)."""
        k = int(np.clip(round((t - tr['t'][0]) / DT), 0, len(tr['t']) - 1))
        i = min(np.searchsorted(wall, t), len(wall) - 1)
        data.qpos[:2] = tr['xy'][k]; data.qpos[2] = tr['z'][k]; data.qpos[3:7] = tr['quat'][k]; data.qpos[7:] = mq[i]
        mujoco.mj_kinematics(model, data)
        return k, np.vstack([data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T for g, v in robot_geoms])

    def wall_clear(body):
        """Per wall: the closest body point, the closest point on the wall, and the distance."""
        out = []
        for b in pair:
            dist, onbox = box_dist(body, b); j = int(np.argmin(dist)); out.append((body[j], onbox[j], float(dist[j])))
        return out

    if a.stills:
        times = t0 + np.array([float(s) for s in a.stills.split(',')]); slowmo = np.zeros(len(times))
    else:
        times, slowmo = timeline(tr, a.fps, a.full, a.slow, event)
    zt = tr['z']; chart_t = (tr['t'][0], tr['t'][-1]) if a.stills else (times[0], times[-1])
    if kind == 'walls':                                         # left: top view, walk left to right, following along it
        uu = np.array([-nrm[1], nrm[0]])                        # left of the walk = up in the image
        win = timeline(tr, a.fps, a.full, a.slow, event)[0]     # (the clip's window also for --stills)
        la = (tr['xy'][(tr['t'] >= win[0]) & (tr['t'] <= win[-1])] - gc) @ uu
        top_ext = 2.8                                           # visible height (m)
        lat_c = float(np.clip((la.max() + la.min()) / 2, -0.4, 0.4)) - 30 * top_ext / PH   # (room for the readout band)
    if kind == 'walls':                                         # the clearance curve for the chart
        ct = np.arange(chart_t[0], chart_t[1], 0.1) if a.stills else times[::2]
        cz = np.array([min(c[2] for c in wall_clear(pose(x)[1])) for x in ct])
        print('closest body point to a wall: %.3f m at t = %.1f s' % (cz.min(), ct[np.argmin(cz)] - t0))
    else:
        sel = (tr['t'] >= chart_t[0]) & (tr['t'] <= chart_t[1]); ct, cz = tr['t'][sel][::8], zt[sel][::8]
    ff = None
    if not a.stills:
        ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (2 * PW, PH + TOP),
                               '-r', str(a.fps), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-crf', '18',
                               '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
    f_title, f_sub, f_lab, f_txt, f_small, f_big = font(30, True), font(19), font(20, True), font(19), font(15), font(26, True)
    title = {'beam': 'Unitree G1, real robot: walking under a virtual beam',
             'walls': 'Unitree G1, real robot: passing a %.2f m gap between virtual walls' % (gap[4] if gap else 0),
             'plain': 'Unitree G1, real robot: walking'}[kind]
    obstacle = {'beam': 'virtual beam (not physical)', 'walls': 'virtual walls (not physical)', 'plain': 'virtual obstacles'}[kind]
    def decorate(sc, trail):
        """Planned path, goal and pelvis trail into a rendered scene."""
        def add(typ, size, pos, rgba, mat=np.eye(3)):
            if sc.ngeom < sc.maxgeom:
                mujoco.mjv_initGeom(sc.geoms[sc.ngeom], typ, np.asarray(size, float), np.asarray(pos, float), mat.flatten(), np.asarray(rgba, np.float32))
                sc.ngeom += 1
                return sc.geoms[sc.ngeom - 1]
        for p0, p1 in zip(path[:-1], path[1:]):
            g = add(mujoco.mjtGeom.mjGEOM_CAPSULE, (0.01, 0, 0), (0, 0, 0), (0.16, 0.75, 0.31, 0.8))
            if g is not None:
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, np.r_[p0[:2], 0.012], np.r_[p1[:2], 0.012])
        if goal is not None:
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, (0.14, 0.004, 0), (goal[0], goal[1], 0.01), (0.16, 0.75, 0.31, 0.7))
        for p in trail:
            add(mujoco.mjtGeom.mjGEOM_SPHERE, (0.022, 0, 0), (p[0], p[1], 0.02), (0.16, 0.45, 0.9, 1))

    trail, ar_cache = [], {}
    for n, t in enumerate(times):
        k, body = pose(t)
        pz = data.qpos[2]
        top = float(body[:, 2].max())
        if a.stills:
            trail = [p for p in tr['xy'][:k + 1:20]]
        elif not trail or np.linalg.norm(trail[-1] - data.qpos[:2]) > 0.05:
            trail.append(data.qpos[:2].copy())
        lk = look[k]

        # left: orthographic. Beam: horizontal side view along the beam (true heights); walls: straight from above
        cam = mujoco.MjvCamera()                                # (the free camera takes orthographic and fovy from vis.global)
        if kind == 'walls':
            top_at = gc + nrm * ((lk - gc) @ nrm) + uu * lat_c
            cam.lookat[:] = (top_at[0], top_at[1], 0); cam.azimuth = math.degrees(side) + 90; cam.elevation = -90; cam.distance = 6.0
            model.vis.global_.orthographic, model.vis.global_.fovy = 1, top_ext
            rend.update_scene(data, cam); decorate(rend.scene, trail)
        else:
            cam.lookat[:] = (lk[0], lk[1], SIDE_ZC); cam.azimuth = math.degrees(side); cam.elevation = 0; cam.distance = 6.0
            model.vis.global_.orthographic, model.vis.global_.fovy = 1, SIDE_EXTENT      # fovy: visible height (m)
            rend.update_scene(data, cam)
        side_img = Image.fromarray(rend.render())
        # view from above
        cam = mujoco.MjvCamera(); cam.lookat[:] = (lk[0], lk[1], 0.45)
        if kind == 'walls':                                     # from behind and aside, into the gap
            cam.azimuth, cam.elevation, cam.distance = math.degrees(side) + 35, -60, 3.8
        else:
            cam.azimuth, cam.elevation, cam.distance = math.degrees(side) - 8, -50, 3.8   # nearly along the obstacle
        model.vis.global_.orthographic, model.vis.global_.fovy = 0, fovy
        rend.update_scene(data, cam); decorate(rend.scene, trail)
        above_img = Image.fromarray(rend.render())

        frame = Image.new('RGB', (2 * PW, PH + TOP), PAPER)
        # left view floor: below z = 0 (edge-on in a horizontal view) paint the ground
        zrow = lambda z: PH / 2 - (z - SIDE_ZC) / SIDE_EXTENT * PH
        g0 = int(round(zrow(0.0))) if kind != 'walls' else PH - 134   # floor row / the readout band
        if kind != 'walls':
            ds = ImageDraw.Draw(side_img); ds.rectangle((0, g0, PW, PH), fill=FLOOR_RGB); ds.line((0, g0, PW, g0), fill=(120, 125, 135), width=2)
        frame.paste(side_img, (0, TOP)); frame.paste(above_img, (PW, TOP))
        ov = Image.new('RGBA', frame.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
        # title band
        dr.text((24, 10), title, font=f_title, fill=INK + (255,))
        dr.text((24, 46), 'scene %s  |  replay of take %s from the recorded joints, IMU and SLAM pose' % (scene or '?', take.name), font=f_sub, fill=(190, 198, 210, 255))
        tb = 't = %4.1f s' % (t - t0); tw = dr.textbbox((0, 0), tb, font=f_big)[2]
        dr.text((2 * PW - 24 - tw, 22), tb, font=f_big, fill=INK + (255,))
        if slowmo[n] > 0.5:
            s = '%.1fx slow motion' % a.slow; sw = dr.textbbox((0, 0), s, font=f_lab)[2]
            dr.rounded_rectangle((2 * PW - 24 - tw - sw - 44, 18, 2 * PW - 24 - tw - 20, 56), 8, fill=ORANGE + (255,))
            dr.text((2 * PW - 24 - tw - sw - 32, 25), s, font=f_lab, fill=INK + (255,))
        dr.line((PW, TOP, PW, TOP + PH), fill=PAPER + (255,), width=4)
        # left view: height ruler, beam-bottom line / gap and clearances, readout
        Y = lambda z: TOP + zrow(z)
        for z in (0.5, 1.0, 1.5) if kind != 'walls' else ():
            dr.line((0, Y(z), 18, Y(z)), fill=(90, 95, 105, 255), width=2)
            dr.text((24, Y(z) - 10), '%.1f m' % z, font=f_small, fill=(90, 95, 105, 255))
        if beam_lo is not None:
            for x in range(70, PW, 24):
                dr.line((x, Y(beam_lo), x + 12, Y(beam_lo)), fill=ORANGE + (230,), width=2)
            dr.text((PW - 262, Y(beam_lo) + 6), 'beam bottom %.2f m' % beam_lo, font=f_lab, fill=(200, 90, 10, 255))
        if kind == 'walls':
            P = lambda p: (PW / 2 + (np.asarray(p)[:2] - top_at) @ nrm * PH / top_ext, PH / 2 - (np.asarray(p)[:2] - top_at) @ uu * PH / top_ext)
            lv = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); dr_main, dr = dr, ImageDraw.Draw(lv)   # clipped to the panel
            dark = (200, 90, 10)
            for b in pair:                                      # the walls, filled (nothing of the robot is above them)
                dr.polygon([P(q) for q in footprint(b, 2)[[0, 1, 3, 2]]], fill=ORANGE + (255,), outline=dark + (255,), width=2)
            off = -nrm * 0.35                                   # gap dimension, just before the walls
            (xa, ya), (xb, yb) = P(ga + off), P(gb + off)
            for e, f in ((ga, ga + off), (gb, gb + off)):
                dr.line((P(e), P(f + off * 0.15)), fill=dark + (255,), width=2)
            dr.line(((xa, ya), (xb, yb)), fill=dark + (255,), width=3)
            for (x0, y0), (x1, y1) in (((xa, ya), (xb, yb)), ((xb, yb), (xa, ya))):
                ux, uy = (x1 - x0), (y1 - y0); ln = math.hypot(ux, uy) or 1; ux, uy = ux / ln, uy / ln
                dr.polygon([(x0, y0), (x0 + 14 * ux - 7 * uy, y0 + 14 * uy + 7 * ux), (x0 + 14 * ux + 7 * uy, y0 + 14 * uy - 7 * ux)], fill=dark + (255,))
            s = 'gap %.2f m' % gw; bw = dr.textbbox((0, 0), s, font=f_lab)
            mx, my = min(xa, xb) - 12 - bw[2], (ya + yb) / 2 - bw[3] / 2
            dr.rounded_rectangle((mx - 6, my - 4, mx + bw[2] + 6, my + bw[3] + 6), 6, fill=(255, 255, 255, 225))
            dr.text((mx, my), s, font=f_lab, fill=dark + (255,))
            sb = PH / top_ext                                   # 1 m scale bar
            dr.rounded_rectangle((PW - 46 - sb, 18, PW - 14, 84), 8, fill=(255, 255, 255, 210))
            dr.line((PW - 30 - sb, 40, PW - 30, 40), fill=(60, 65, 75, 255), width=4)
            for x in (PW - 30 - sb, PW - 30):
                dr.line((x, 32, x, 48), fill=(60, 65, 75, 255), width=3)
            dr.text((PW - 30 - sb / 2 - 14, 50), '1 m', font=f_lab, fill=(60, 65, 75, 255))
            clears = wall_clear(body); lines = []
            for (pb, pw, dist), name in sorted(zip(clears, names), key=lambda c: c[1]):
                col = (200, 30, 30) if dist <= 0 else (230, 120, 10) if dist < 0.05 else (30, 150, 60)
                if abs((data.qpos[:2] - gc) @ nrm) < 0.45:      # in the passage: the shortest segment robot -> wall
                    p1, p2 = P(pb), P(pw)
                    dr.line((p1, p2), fill=col + (255,), width=3)
                    dr.ellipse((p1[0] - 5, p1[1] - 5, p1[0] + 5, p1[1] + 5), fill=col + (255,), outline=INK + (255,))
                    lab = '%.0f cm' % (dist * 100); lw = dr.textbbox((0, 0), lab, font=f_lab)[2]
                    lx_, ly_ = p2[0] + 14, p2[1] - 13 + (-18 if p2[1] < p1[1] else 18)
                    dr.rounded_rectangle((lx_ - 6, ly_ - 2, lx_ + lw + 6, ly_ + 26), 6, fill=(255, 255, 255, 230))
                    dr.text((lx_, ly_), lab, font=f_lab, fill=col + (255,))
                lines.append('%s %s' % (name, ('%.0f cm' % (dist * 100)) if dist < 1 else '%.2f m' % dist))
            close = min(c[2] for c in clears)
            ov.paste(Image.alpha_composite(ov.crop((0, TOP, PW, TOP + PH)), lv), (0, TOP)); dr = dr_main
            plate(dr, (16, TOP + 14), ['TOP VIEW', 'orthographic, true metres'], [f_lab, f_small])
            plate(dr, (16, TOP + g0 + 14), ['closest to a wall %s' % (('%.1f cm' % (close * 100)) if close < 1 else '%.2f m' % close),
                                            '   '.join(lines)], [f_lab, f_txt])
            chart_label, zlo, zhi, cur = 'closest body point to a wall', 0.0, 0.5, close
        else:
            plate(dr, (16, TOP + 14), ['SIDE VIEW', 'orthographic, true heights'], [f_lab, f_small])
            lines = ['pelvis %.2f m' % pz, 'head top %.2f m' % top]
            for b in boxes:                                     # the highest body point inside the box's footprint
                loc = (body - b['p']) @ rot(b['q']); inside = (np.abs(loc[:, :2]) < b['scale'][:2] / 2).all(1)
                if inside.any():
                    clr = b['p'][2] - b['scale'][2] / 2 - body[inside, 2].max()
                    lines[1] += '   under the beam: %s %.2f m' % ('clearance' if clr >= 0 else 'INSIDE by', abs(clr))
                    break
            plate(dr, (16, TOP + g0 + 14), lines, [f_lab, f_txt])
            chart_label, zlo, zhi, cur = 'pelvis height over the clip', 0.45, 0.85, pz
        # strip chart in the floor band
        cx0, cx1, cy0, cy1 = PW - 310, PW - 26, TOP + g0 + 40, TOP + PH - 16
        dr.rounded_rectangle((cx0 - 12, cy0 - 30, cx1 + 12, cy1 + 8), 8, fill=PAPER + (185,))
        dr.text((cx0, cy0 - 26), chart_label, font=f_small, fill=INK + (255,))
        X = lambda tt: cx0 + (tt - chart_t[0]) / max(chart_t[1] - chart_t[0], 1e-6) * (cx1 - cx0)
        Z = lambda z: cy1 - (np.clip(z, zlo, zhi) - zlo) / (zhi - zlo) * (cy1 - cy0)
        dr.line([(X(a_), Z(b_)) for a_, b_ in zip(ct, cz)], fill=(150, 200, 255, 255), width=3)
        dr.ellipse((X(t) - 6, Z(cur) - 6, X(t) + 6, Z(cur) + 6), fill=ORANGE + (255,), outline=INK + (255,))
        # view from above: label, legend, AR inset
        plate(dr, (PW + 16, TOP + 14), ['VIEW FROM ABOVE'], [f_lab])
        lx, ly = PW + 16, TOP + 66
        items = [(ORANGE, obstacle), ((140, 145, 158), 'room, SLAM map (floor plan)'),
                 (GREEN, 'planned path, goal'), (BLUE, 'pelvis trail')]
        dr.rounded_rectangle((lx, ly, lx + 300, ly + 20 + 28 * len(items)), 8, fill=PAPER + (170,))
        for j, (c, s) in enumerate(items):
            dr.rounded_rectangle((lx + 12, ly + 14 + 28 * j, lx + 30, ly + 30 + 28 * j), 3, fill=c + (255,))
            dr.text((lx + 40, ly + 12 + 28 * j), s, font=f_small, fill=INK + (255,))
        j = latest(frames, t)
        if j >= 0:
            if j not in ar_cache:
                ar_cache.clear(); ar_cache[j] = Image.open(io.BytesIO(read_compressed(frames[j][1]))).convert('RGB').resize((432, 243), Image.LANCZOS)
            ax, ay = 2 * PW - 432 - 18, TOP + PH - 243 - 18
            dr.rectangle((ax - 3, ay - 34, ax + 432 + 3, ay + 243 + 3), fill=PAPER + (230,))
            dr.text((ax + 8, ay - 28), 'head camera with the AR overlay', font=f_small, fill=INK + (255,))
        frame = Image.alpha_composite(frame.convert('RGBA'), ov).convert('RGB')
        if j >= 0:
            frame.paste(ar_cache[j], (ax, ay))
        if a.stills:
            p = out.with_name('%s_t%04.1f.png' % (out.stem, t - t0)); frame.save(p); print('wrote', p)
        else:
            ff.stdin.write(frame.tobytes())
    if ff:
        ff.stdin.close(); ff.wait()
        print('wrote %s: %d frames, %.1f s (take %.1f..%.1f s)' % (out, len(times), len(times) / a.fps, times[0] - t0, times[-1] - t0))


if __name__ == '__main__':
    main()
