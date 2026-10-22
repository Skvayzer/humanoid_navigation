#!/usr/bin/env python3
"""Presentation video from a demo take: the AR camera next to a top-down view of how the robot sees the world.

  python3 render_take.py <take_dir> [--fps 10] [--out <take_dir>/demo.mp4]

Reads <take_dir>/bag/*.db3 (ros2 bag, sqlite3) with no ROS install: a small CDR reader for the few message types the
demo records. Top-down (map frame): the room from SLAM (grey), the virtual boxes (orange), route and geofence, the
policy's body sites coloured by their distance to the virtual obstacles, the pelvis trail and heading, the goal.
Needs numpy + OpenCV (+ ffmpeg for an H.264 file).
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


from bagread import Cdr, latest, load, read_cloud, read_compressed, read_markers, read_pose_stamped, read_string, read_tf, rot, yaw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('take'); ap.add_argument('--fps', type=float, default=10.); ap.add_argument('--out')
    a = ap.parse_args(); take = Path(a.take); out = Path(a.out) if a.out else take / 'demo.mp4'
    m = load(take)
    t_map_ci = np.eye(4)
    for _, raw in m.get('/tf_static', []):
        for parent, child, t, q in read_tf(raw):
            if parent == 'map' and child == 'camera_init':
                t_map_ci[:3, :3] = rot(q); t_map_ci[:3, 3] = t
    pelvis = [(ts, *read_pose_stamped(raw)) for ts, raw in m.get('/g1_wb/pelvis', [])]
    goals = [(ts, read_pose_stamped(raw)[0]) for ts, raw in m.get('/g1_wb/goal', [])]
    status = [(ts, json.loads(read_string(raw))) for ts, raw in m.get('/g1_wb/status', [])]
    scenes = [(ts, read_markers(raw)) for ts, raw in m.get('/g1_wb/virtual_scene', [])]
    maps = m.get('/g1_slam/map', []); sites = m.get('/g1_wb/sites', []); frames = m.get('/g1_wb/ar/compressed', [])
    if not frames:
        raise SystemExit('no AR frames in %s' % take)
    # fixed top-down window: the trail, the virtual scene and 1.5 m round them
    xy = np.array([p[:2] for _, p, _ in pelvis]) if pelvis else np.zeros((1, 2))
    if scenes:
        for mk in scenes[-1][1]:
            xy = np.vstack([xy, mk['p'][:2][None]] + ([mk['points'][:, :2]] if len(mk['points']) and mk['ns'] == 'path' else []))
    lo, hi = xy.min(0) - 1.5, xy.max(0) + 1.5; span = max(hi - lo); c = (lo + hi) / 2; lo, hi = c - span / 2, c + span / 2
    S = 720; px = lambda p: (int((p[0] - lo[0]) / span * S), int(S - (p[1] - lo[1]) / span * S))
    cam_w, cam_h = 1280, 720
    t0, t1 = frames[0][0], frames[-1][0]; n = int((t1 - t0) * a.fps) + 1
    tmp = out.with_suffix('.raw.mp4'); vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (cam_w + S, S))
    cached_map = (None, None)
    for k in range(n):
        t = t0 + k / a.fps
        i = latest(frames, t); cam = cv2.imdecode(np.frombuffer(read_compressed(frames[i][1]), np.uint8), cv2.IMREAD_COLOR)
        cam = cv2.resize(cam, (cam_w, cam_h))
        top = np.full((S, S, 3), 250, np.uint8)
        for g in np.arange(math.floor(lo[0]), hi[0] + 1):                    # 1 m grid
            cv2.line(top, px((g, lo[1])), px((g, hi[1])), (232, 232, 232), 1)
        for g in np.arange(math.floor(lo[1]), hi[1] + 1):
            cv2.line(top, px((lo[0], g)), px((hi[0], g)), (232, 232, 232), 1)
        j = latest(maps, t)
        if j >= 0:
            if cached_map[0] != j:
                pts, _ = read_cloud(maps[j][1]); pts = pts @ t_map_ci[:3, :3].T + t_map_ci[:3, 3]
                cached_map = (j, pts[(pts[:, 2] > 0.10) & (pts[:, 2] < 1.8)])
            for p in cached_map[1]:
                q = px(p)
                if 0 <= q[0] < S and 0 <= q[1] < S:
                    top[q[1], q[0]] = (110, 110, 110)
        j = latest(scenes, t)
        if j >= 0:
            over = top.copy()
            for mk in scenes[j][1]:
                if mk['ns'] == 'virtual_scene':
                    h = mk['scale'][:2] / 2; R = rot(mk['q'])[:2, :2]
                    corners = np.array([[sx * h[0], sy * h[1]] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) @ R.T + mk['p'][:2]
                    poly = np.array([px(cc) for cc in corners], np.int32)
                    cv2.fillConvexPoly(over, poly, (26, 115, 255)); cv2.polylines(top, [poly], True, (0, 90, 255), 2)
                elif mk['ns'] in ('path', 'geofence') and len(mk['points']) > 1:
                    cv2.polylines(top, [np.array([px(p) for p in mk['points']], np.int32)], mk['ns'] == 'geofence',
                                  (80, 200, 60) if mk['ns'] == 'path' else (40, 40, 230), 2, cv2.LINE_AA)
            top[:] = cv2.addWeighted(over, 0.45, top, 0.55, 0)
        trail = [px(p) for ts, p, _ in pelvis if ts <= t]
        if len(trail) > 1:
            cv2.polylines(top, [np.array(trail, np.int32)], False, (200, 120, 30), 2, cv2.LINE_AA)
        j = latest(goals, t)
        if j >= 0:
            cv2.drawMarker(top, px(goals[j][1]), (40, 170, 40), cv2.MARKER_TILTED_CROSS, 18, 3)
        j = latest(sites, t)
        if j >= 0:
            pts, sdf = read_cloud(sites[j][1])
            for p, d in zip(pts, sdf if sdf is not None else np.ones(len(pts))):
                f = float(np.clip(d / 0.3, 0, 1)); cv2.circle(top, px(p), 4, (int(60 * f), int(60 + 160 * f), int(230 * (1 - f) + 40)), -1, cv2.LINE_AA)
        j = latest(pelvis, t)
        if j >= 0:
            p, q = pelvis[j][1], pelvis[j][2]; h = yaw(q)
            cv2.circle(top, px(p), 9, (200, 120, 30), -1, cv2.LINE_AA)
            cv2.arrowedLine(top, px(p), px(p[:2] + 0.5 * np.array([math.cos(h), math.sin(h)])), (200, 120, 30), 3, cv2.LINE_AA, tipLength=0.3)
        j = latest(status, t); st = status[j][1] if j >= 0 else {}
        label = 't %4.1f s   %s   %s   obstacles: %s' % (t - t0, st.get('mode', '?'), st.get('scene', ''), st.get('obstacles', '?'))
        cv2.putText(top, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(top, 'top view (map), 1 m grid: room (grey), virtual obstacles (orange), route, trail, goal',
                    (10, S - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
        frame = np.zeros((S, cam_w + S, 3), np.uint8); frame[:cam_h, :cam_w] = cam; frame[:, cam_w:] = top
        cv2.putText(frame, 'head camera + virtual obstacles (AR)', (12, cam_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    try:                                                                     # H.264 plays everywhere (slides, browsers)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(out)], check=True)
        tmp.unlink()
    except Exception:
        tmp.rename(out)
    print('wrote %s: %d frames, %.1f s, %d AR frames in the bag' % (out, n, n / a.fps, len(frames)))


if __name__ == '__main__':
    main()
