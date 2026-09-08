"""ROS-independent geometry and bounded CAT obstacle-input representation.

No policy imports, shared memory, robot SDK, goals, or command messages.
"""
import ctypes as ct
from pathlib import Path

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
        self.lib.cat_insert.argtypes = [ct.c_void_p, floatp, bytep, ct.c_size_t, floatp, ct.c_double]
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

    def insert(self, xyz, hits, sensor_origin, max_range=2.5):
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        hits = np.ascontiguousarray(hits, dtype=np.uint8)
        sensor_origin = np.ascontiguousarray(sensor_origin, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or hits.shape != (len(xyz),) or sensor_origin.shape != (3,):
            raise ValueError("bad integration array shapes")
        self._check(self.lib.cat_insert(self.handle, xyz, hits, len(xyz), sensor_origin, max_range))

    def export(self, origin):
        origin = np.ascontiguousarray(origin, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError("bad grid origin")
        result = np.zeros(SHAPE, dtype=np.uint8)
        self._check(self.lib.cat_export(self.handle, origin, *SHAPE, result, result.size))
        return result
