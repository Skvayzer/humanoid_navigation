"""Read a demo take's ros2 bag (sqlite3) without ROS: a small little-endian CDR reader for the message types the
demo recorder writes. numpy only (used by render_take.py and replay_take.py)."""
import glob
import math
import sqlite3
import struct
from pathlib import Path

import numpy as np


class Cdr:
    """Little-endian CDR reader; alignment counts from the payload start (after the 4-byte encapsulation)."""

    def __init__(self, raw):
        self.b = bytes(raw)[4:]; self.o = 0

    def align(self, n):
        self.o = (self.o + n - 1) & ~(n - 1)

    def take(self, fmt, n):
        self.align(n); v = struct.unpack_from('<' + fmt, self.b, self.o); self.o += struct.calcsize('<' + fmt)
        return v

    def u32(self): return self.take('I', 4)[0]
    def i32(self): return self.take('i', 4)[0]
    def f64(self, k=1): v = self.take('%dd' % k, 8); return v if k > 1 else v[0]
    def f32(self, k=1): v = self.take('%df' % k, 4); return v if k > 1 else v[0]
    def u8(self): v = self.b[self.o]; self.o += 1; return v

    def string(self):
        n = self.u32(); s = self.b[self.o:self.o + n - 1].decode('utf-8', 'replace'); self.o += n
        return s

    def octets(self):
        n = self.u32(); s = self.b[self.o:self.o + n]; self.o += n
        return s

    def header(self):
        sec, nsec = self.take('iI', 4)
        return sec + 1e-9 * nsec, self.string()


def pose(r):
    return np.array(r.f64(3)), np.array(r.f64(4))                       # position, quaternion x y z w


def read_pose_stamped(raw):
    r = Cdr(raw); r.header(); p, q = pose(r)
    return p, q


def read_compressed(raw):
    r = Cdr(raw); r.header(); r.string()
    return r.octets()


def read_string(raw):
    return Cdr(raw).string()


def read_cloud(raw):
    r = Cdr(raw); r.header(); h, w = r.u32(), r.u32()
    fields = {}
    for _ in range(r.u32()):
        name = r.string(); off = r.u32(); r.u8(); r.u32(); fields[name] = off
    r.u8(); step = r.u32(); r.u32(); data = r.octets()
    n = h * w; a = np.frombuffer(data, np.uint8).reshape(n, step) if n else np.zeros((0, step), np.uint8)
    col = lambda k: a[:, fields[k]:fields[k] + 4].copy().view('<f4')[:, 0]
    out = np.column_stack([col('x'), col('y'), col('z')]) if n else np.zeros((0, 3))
    extra = next((k for k in fields if k not in 'xyz'), None)
    return out, (col(extra) if n and extra else None)


def read_markers(raw):
    r = Cdr(raw); out = []
    for _ in range(r.u32()):
        _, frame = r.header(); ns = r.string(); mid = r.i32(); typ = r.i32(); act = r.i32()
        p, q = pose(r); scale = np.array(r.f64(3)); color = r.f32(4); r.take('iI', 4); r.u8()
        pts = [r.f64(3) for _ in range(r.u32())]
        for _ in range(r.u32()):
            r.f32(4)
        r.string(); r.string(); r.u8()
        out.append(dict(ns=ns, type=typ, action=act, p=p, q=q, scale=scale, color=color, points=np.array(pts).reshape(-1, 3)))
    return out


def read_tf(raw):
    r = Cdr(raw); out = []
    for _ in range(r.u32()):
        _, parent = r.header(); child = r.string(); t = np.array(r.f64(3)); q = np.array(r.f64(4))
        out.append((parent, child, t, q))
    return out


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load(take):
    db = glob.glob(str(Path(take) / 'bag' / '*.db3'))[0]; c = sqlite3.connect(db)
    names = {tid: name for tid, name in c.execute('select id, name from topics')}
    msgs = {}
    for tid, ts, data in c.execute('select topic_id, timestamp, data from messages order by timestamp'):
        msgs.setdefault(names[tid], []).append((ts * 1e-9, data))
    return msgs


def latest(series, t):
    """Index of the last message at or before t (or -1)."""
    lo, hi = 0, len(series)
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1
