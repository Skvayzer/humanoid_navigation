"""ROS-independent geometry and bounded CAT obstacle-input representation.

No policy imports, shared memory, robot SDK, goals, or command messages.
"""
import ctypes as ct
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import struct

import numpy as np
from scipy.ndimage import binary_closing, convolve

SHAPE = (128, 128, 35)
RESOLUTION = 0.04


def cat_preprocess(occupied):
    """CAT's real-deployment morphology, isolated from its motion/SHM imports.

    Adapted from GalaxyGeneralRobotics/Click-and-Traverse (Apache-2.0),
    deploy/scripts/exp_dis_pf/octomap_bridge.py at 866ba392f1c1e84b92ad75fa66550f26e8af8e48.
    Changes: pure function, explicit input shape, copy, no dummy obstacles.
    See upstream/LICENSE and upstream/NOTICE.md. This can REMOVE thin obstacles
    and ADD filled volumes; visualize it alongside the unmodified occupancy.
    """
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.shape != SHAPE:
        raise ValueError("CAT expects XYZ shape (128, 128, 35)")
    result = binary_closing(occupied, structure=np.ones((5, 5, 5), dtype=bool),
                            iterations=2)
    neighbors = convolve(result.astype(np.int32), np.ones((3, 3, 3)),
                         mode="constant", cval=0)
    result &= neighbors > 1
    result[:, :, 20:] = np.cumsum(result[:, :, 20:], axis=2) > 0
    result[:, :, :5] = (np.cumsum(result[:, :, :5][:, :, ::-1], axis=2) > 0)[:, :, ::-1]
    return result


def rotation_matrix(xyzw):
    q = np.asarray(xyzw, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
        raise ValueError("invalid transform quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform_points(points, translation, xyzw):
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("invalid translation")
    return np.asarray(np.asarray(points) @ rotation_matrix(xyzw).T + translation,
                      dtype=np.float32)


def cloud_xyz(message, max_points=20000):
    """Decode PointCloud2 with field offsets, endian, row padding and NaNs."""
    if max_points < 1 or message.point_step < 12:
        raise ValueError("invalid cloud point_step or point limit")
    fields = {f.name: f for f in message.fields}
    offsets, formats = [], []
    for name in ("x", "y", "z"):
        f = fields.get(name)
        if f is None or f.count != 1 or f.datatype not in (7, 8):
            raise ValueError("cloud must contain float x/y/z")
        size = 4 if f.datatype == 7 else 8
        if f.offset < 0 or f.offset + size > message.point_step:
            raise ValueError("field outside point_step")
        offsets.append(f.offset)
        formats.append((">" if message.is_bigendian else "<") + "f" + str(size))
    if message.width == 0 or message.height == 0:
        return np.empty((0, 3), dtype=np.float32)
    if message.row_step < message.width * message.point_step:
        raise ValueError("cloud row_step too small")
    data = memoryview(bytes(message.data))
    if len(data) < message.row_step * message.height:
        raise ValueError("cloud data truncated")
    dtype = np.dtype(dict(names=["x", "y", "z"], formats=formats,
                          offsets=offsets, itemsize=message.point_step))
    records = np.ndarray((message.height, message.width), dtype=dtype,
                         buffer=data, strides=(message.row_step, message.point_step))
    # Sample deterministically across the whole scan before materializing XYZ.
    count = message.width * message.height
    indices = np.arange(0, count, max(1, int(np.ceil(count / max_points))))
    rows, cols = indices // message.width, indices % message.width
    xyz = np.column_stack([records[name][rows, cols] for name in ("x", "y", "z")])
    return np.ascontiguousarray(xyz[np.isfinite(xyz).all(axis=1)], dtype=np.float32)


def grid_origin(position, floor_z=0.0):
    # Map-axis aligned, floor-relative (NOT centered on body Z, NOT yaw rotated).
    origin = np.floor(np.asarray(position, dtype=float) / RESOLUTION) * RESOLUTION
    origin[:2] -= np.array(SHAPE[:2]) * RESOLUTION / 2
    origin[2] = floor_z
    return origin


def centers(mask, origin):
    return np.asarray(np.argwhere(mask) * RESOLUTION + np.asarray(origin) + RESOLUTION/2,
                      dtype=np.float32)


@dataclass(frozen=True)
class Scan:
    message: object
    received: float
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class LivoxPacket:
    points: np.ndarray
    point_num: int
    frame: str
    start_ns: int
    end_ns: int


def decode_livox_cdr(data):
    """Strict CDR1 decoder for the bundled Livox CustomMsg definition.

    rclpy raw=True avoids allocating ~20,000 Python ROS objects every 100 ms.
    This changes only CAT's subscription, not messages, middleware or drivers.
    All alignment is relative to the payload after its 4-byte encapsulation.
    """
    if len(data) < 24 or data[:2] not in (b'\x00\x00', b'\x00\x01'):
        raise ValueError("unsupported/truncated Livox CDR1 encapsulation")
    endian = '<' if data[1] == 1 else '>'
    sec, nsec, length = struct.unpack_from(endian+'iII', data, 4)
    if sec < 0 or nsec >= 1000000000 or not 1 <= length <= 256 or 16+length > len(data):
        raise ValueError("invalid Livox CDR header")
    if data[15+length] != 0:
        raise ValueError("unterminated Livox frame")
    frame = data[16:15+length].decode('utf-8')
    pos = 4 + ((12+length+7)//8)*8
    if pos+20 > len(data):
        raise ValueError("truncated Livox metadata")
    timebase, count = struct.unpack_from(endian+'QI', data, pos)
    sequence_count = struct.unpack_from(endian+'I', data, pos+16)[0]
    pos += 20
    if count != sequence_count or not 0 < count <= 100000:
        raise ValueError("invalid Livox point count")
    # CDR may omit the final structure's single byte of trailing alignment.
    if not pos + count*20 - 1 <= len(data) <= pos + count*20:
        raise ValueError("Livox CDR point sequence size mismatch")
    if len(data) == pos+count*20-1:
        data = bytes(data)+b'\x00'
    dtype = np.dtype(dict(names=['offset_time','x','y','z','reflectivity','tag','line'],
        formats=[endian+'u4',endian+'f4',endian+'f4',endian+'f4','u1','u1','u1'],
        offsets=[0,4,8,12,16,17,18], itemsize=20))
    points = np.frombuffer(data, dtype=dtype, count=count, offset=pos)
    span = int(points['offset_time'].max())
    start = sec*1000000000+nsec
    if start <= 0 or abs(timebase-start) > 1000000 or span > 200000000:
        raise ValueError("invalid Livox point times/timebase")
    return LivoxPacket(points, count, frame, start, start+span)


class ScanBuffer:
    """Small thread-safe queue; processing picks newest scan fully covered by TF."""
    def __init__(self, capacity=16):
        self.queue = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.generation = 0

    def append(self, scan):
        with self.lock:
            if self.queue and scan.start_ns < self.queue[-1].start_ns:
                self.queue.clear()
                self.generation += 1
            if not self.queue or scan.start_ns > self.queue[-1].start_ns:
                self.queue.append(scan)

    def snapshot(self):
        with self.lock:
            return list(self.queue), self.generation

    @staticmethod
    def select(scans, now, max_age, after_ns, tf_end_ns):
        return next((s for s in reversed(scans)
                     if s.start_ns > after_ns and now-s.received <= max_age
                     and s.end_ns <= tf_end_ns), None)


def livox_points(message, max_points=20000, min_range=0.0):
    """Raw Livox returns. Retain close points before sampling farther points.

    No software blind zone by default, but nonfinite/zero/invalid-tag returns
    never become obstacles OR clearing rays. Returns nanosecond point offsets.
    """
    count = len(message.points)
    if count != message.point_num or not 0 < count <= 100000 or max_points < 1:
        raise ValueError("malformed or oversized Livox packet")
    if isinstance(message, LivoxPacket):
        values = np.column_stack([message.points[n] for n in ('x','y','z','offset_time','tag','line')])
    else:
        values = np.array([(p.x, p.y, p.z, p.offset_time, p.tag, p.line)
                           for p in message.points], dtype=np.float64)
    xyz, offsets = values[:, :3], values[:, 3].astype(np.int64)
    if offsets.min() < 0 or offsets.max() > 200000000:
        raise ValueError("invalid Livox point times (>200 ms)")
    ranges = np.linalg.norm(xyz, axis=1)
    tags = values[:, 4].astype(np.uint8) & 0x30
    valid = np.isfinite(xyz).all(axis=1) & (ranges > max(min_range, 1e-6))
    valid &= (values[:, 5] < 4) & ((tags == 0) | (tags == 0x10))
    near = np.flatnonzero(valid & (ranges < 0.5))
    far = np.flatnonzero(valid & (ranges >= 0.5))
    def sample(indices, limit):
        if len(indices) <= limit:
            return indices
        return indices[np.linspace(0, len(indices)-1, limit).astype(int)]
    kept_near = sample(near, min(max_points, len(near)))
    kept = np.concatenate((kept_near, sample(far, max_points-len(kept_near))))
    info = dict(input_points=count, valid_points=int(valid.sum()),
                invalid_returns=int((~valid).sum()), nearfield_available=len(near),
                nearfield_retained=len(kept_near), sampled_points=len(kept))
    return np.asarray(xyz[kept], dtype=np.float32), offsets[kept], info


def deskew_livox(xyz, offsets_ns, start_ns, pose_at, sensor_translation,
                 bin_ns=10000000):
    """Use full SLAM TF at each 10 ms point-time bin (bounded pose interpolation).

    G1 vendor cloud is roll-flipped relative to its native IMU. Apply exactly
    the existing FAST-LIO extrinsic_R=diag(1,-1,-1), then extrinsic_T. The pose
    callback must provide sensor-time map<-body TF; no latest-pose fallback.
    Return a distinct world ray origin per point as the LiDAR moves.
    """
    xyz = np.asarray(xyz)
    offsets_ns = np.asarray(offsets_ns, dtype=np.int64)
    if xyz.shape != (len(offsets_ns), 3) or bin_ns < 1:
        raise ValueError("bad timed-point arrays")
    body = xyz * np.array([1, -1, -1]) + np.asarray(sensor_translation)
    points = np.empty_like(xyz, dtype=np.float32)
    origins = np.empty_like(xyz, dtype=np.float32)
    for group in np.unique(offsets_ns // bin_ns):
        mask = offsets_ns // bin_ns == group
        stamp_ns = start_ns + int(np.mean(offsets_ns[mask]))
        translation, quat = pose_at(stamp_ns)
        points[mask] = transform_points(body[mask], translation, quat)
        origins[mask] = transform_points([sensor_translation], translation, quat)[0]
    return points, origins


class OccupancyMap:
    def __init__(self, library=None):
        self.lib = ct.CDLL(str(library or Path(__file__).parent / "build/libcat_octomap.so"))
        floatp = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        doublep = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        bytep = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        self.lib.cat_create.argtypes = [ct.c_double]
        self.lib.cat_create.restype = ct.c_void_p
        self.lib.cat_destroy.argtypes = [ct.c_void_p]
        self.lib.cat_reset.argtypes = [ct.c_void_p]
        self.lib.cat_size.argtypes = [ct.c_void_p]
        self.lib.cat_size.restype = ct.c_size_t
        self.lib.cat_error.restype = ct.c_char_p
        self.lib.cat_insert.argtypes = [ct.c_void_p, floatp, bytep, ct.c_size_t, floatp,
                                       ct.c_size_t, ct.c_double, ct.c_double]
        self.lib.cat_prune.argtypes = [ct.c_void_p, doublep, doublep, ct.c_double, ct.c_double]
        self.lib.cat_export.argtypes = [ct.c_void_p, doublep, ct.c_int, ct.c_int, ct.c_int,
                                       bytep, ct.c_size_t]
        self.handle = self.lib.cat_create(RESOLUTION)
        if not self.handle:
            raise RuntimeError("cannot create OctoMap")

    def close(self):
        if self.handle:
            self.lib.cat_destroy(self.handle)
            self.handle = None

    def reset(self):
        self._check(self.lib.cat_reset(self.handle))

    @property
    def nodes(self):
        return self.lib.cat_size(self.handle)

    def _check(self, result):
        if result != 0:
            raise RuntimeError("OctoMap: " + self.lib.cat_error().decode())

    def insert(self, xyz, hits, sensor_origin, max_range=2.5, observed_at=None):
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        hits = np.ascontiguousarray(hits, dtype=np.uint8)
        sensor_origin = np.ascontiguousarray(sensor_origin, dtype=np.float32)
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or hits.shape != (len(xyz),)
                or sensor_origin.shape not in ((3,), (len(xyz), 3))):
            raise ValueError("bad integration array shapes")
        self._check(self.lib.cat_insert(self.handle, xyz, hits, len(xyz), sensor_origin,
                                       1 if sensor_origin.ndim == 1 else len(xyz), max_range,
                                       time.monotonic() if observed_at is None else observed_at))

    def prune(self, origin, now, ttl=30.0):
        origin = np.ascontiguousarray(origin, dtype=np.float64)
        upper = np.ascontiguousarray(origin + np.asarray(SHAPE)*RESOLUTION, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError("bad prune origin")
        result = self.lib.cat_prune(self.handle, origin, upper, now, ttl)
        if result < 0:
            self._check(result)
        return result

    def export(self, origin):
        origin = np.ascontiguousarray(origin, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError("bad grid origin")
        result = np.zeros(SHAPE, dtype=np.uint8)
        self._check(self.lib.cat_export(self.handle, origin, *SHAPE, result, result.size))
        return result
