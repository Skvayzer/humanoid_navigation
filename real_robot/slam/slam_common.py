"""Shared helpers of the SLAM tools (numpy only, Python 3.8 / numpy 1.17 compatible)."""
import math
import struct

import numpy as np

# FAST_LIO_G1 config (real_robot/config/fastlio_g1.yaml): native vendor cloud, p_body = p_lidar + T_EXT
T_EXT = np.array([-0.011, -0.02329, 0.04412])


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w); x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(r):
    """-> (x, y, z, w)"""
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(r))); j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        q[i] = 0.25 * s; q[j] = (r[j, i] + r[i, j]) / s; q[k] = (r[k, i] + r[i, k]) / s
    q = np.asarray(q); return q / np.linalg.norm(q)


def pose_to_matrix(position, orientation):
    t = np.eye(4); t[:3, :3] = quat_to_matrix(orientation.x, orientation.y, orientation.z, orientation.w)
    t[:3, 3] = [position.x, position.y, position.z]; return t


def inverse(t):
    out = np.eye(4); out[:3, :3] = t[:3, :3].T; out[:3, 3] = -t[:3, :3].T @ t[:3, 3]; return out


def cloud_xyz(msg, fields=('x', 'y', 'z')):
    """PointCloud2 -> float64 [N, 3] for float32 x/y/z fields at any offsets."""
    offs = {f.name: f.offset for f in msg.fields}
    dtype = np.dtype({'names': list(fields), 'formats': ['<f4'] * len(fields), 'offsets': [offs[n] for n in fields],
                      'itemsize': msg.point_step})
    a = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    out = np.column_stack([a[n].astype(np.float64) for n in fields])
    return out[np.isfinite(out).all(1)]


def gravity_frame(up_body):
    """Rotation map <- body with map z = up, map x = horizontal projection of body x."""
    up = up_body / np.linalg.norm(up_body)
    x = np.array([1.0, 0.0, 0.0]) - up * up[0]; x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(up, x), up])


def fit_floor(points_map, z_guess=0.0, band=0.3, r_min=0.6, r_max=6.0, origin_xy=(0.0, 0.0)):
    """Dominant horizontal plane near z_guess: returns (a, b, c, inliers, rms) for z = a x + b y + c."""
    p = points_map
    r = np.hypot(p[:, 0] - origin_xy[0], p[:, 1] - origin_xy[1])
    sel = (r > r_min) & (r < r_max) & (np.abs(p[:, 2] - z_guess) < band)
    if sel.sum() < 500:
        raise ValueError('too few floor candidates (%d)' % sel.sum())
    hist, edges = np.histogram(p[sel, 2], bins=np.arange(z_guess - band, z_guess + band + 1e-9, 0.02))
    z0 = edges[int(np.argmax(hist))] + 0.01
    band_sel = sel & (np.abs(p[:, 2] - z0) < 0.05)
    A = np.column_stack([p[band_sel, 0], p[band_sel, 1], np.ones(band_sel.sum())])
    coef, _, _, _ = np.linalg.lstsq(A, p[band_sel, 2], rcond=None)
    rms = float(np.std(p[band_sel, 2] - A @ coef))
    return float(coef[0]), float(coef[1]), float(coef[2]), int(band_sel.sum()), rms


def level_transform(a, b, c):
    """Rigid transform that makes the plane z = a x + b y + c the plane z = 0 (small rotation + shift)."""
    n = np.array([-a, -b, 1.0]); n /= np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(n, z); s = np.linalg.norm(v); cth = n @ z
    if s < 1e-12:
        r = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        r = np.eye(3) + vx + vx @ vx * ((1 - cth) / s ** 2)
    t = np.eye(4); t[:3, :3] = r
    t[2, 3] = -(r @ np.array([0.0, 0.0, c]))[2]
    return t


def write_pcd(path, xyz, intensity=None):
    xyz = np.asarray(xyz, np.float32)
    inten = np.zeros(len(xyz), np.float32) if intensity is None else np.asarray(intensity, np.float32)
    data = np.column_stack([xyz, inten]).astype('<f4')
    header = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
              'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n' % (len(data), len(data)))
    with open(path, 'wb') as f:
        f.write(header.encode('ascii')); f.write(data.tobytes())


def read_pcd(path):
    with open(path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('ascii').strip()
            key, _, value = line.partition(' ')
            header[key] = value
            if key == 'DATA':
                break
        fields = header['FIELDS'].split(); n = int(header['POINTS'])
        if header['DATA'] != 'binary' or any(s != '4' for s in header['SIZE'].split()):
            raise ValueError('only binary float32 PCD supported')
        data = np.frombuffer(f.read(4 * len(fields) * n), '<f4').reshape(n, len(fields))
    return {name: data[:, i] for i, name in enumerate(fields)}
