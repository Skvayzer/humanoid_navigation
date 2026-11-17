#!/usr/bin/env python3
"""Convert a dataset bag into a .g1replay file, the input of fastlio_mapping --replay.

The lidar is rewritten in the exact layout of the G1 vendor stream (/utlidar/cloud_livox_mid360),
so FAST_LIO reads every dataset through the same code path as the robot (lidar_type 2):
    x, y, z, intensity: float32 @0/4/8/12   ring: uint16 @16   time: float32 ns since scan start @18
    point_step 22, header.stamp = scan start, all points kept (including (0,0,0) no-returns)
Only the Mid-360 lidar and its own IMU are kept. Ground truth, if any (a pose topic in the bag or a
TUM file shipped with the dataset), is written as <name>.gt.tum next to it.

File format (little endian):
    b'G1RP' | u32 version=1 | u32 meta_len | meta JSON
    records sorted by stamp, each starting with u8 type and i64 stamp_ns:
      type 1 IMU:   6 x f64 (acc xyz [as recorded], gyro xyz [rad/s])
      type 2 CLOUD: i64 end_ns | u32 n_points | u32 zlib_len | zlib(n_points x 22 bytes)

Usage (Mac, needs numpy + rosbags):
    python make_replay.py <dataset> <bag> <out_dir> [--gt <gt.txt>]
    e.g. python make_replay.py tiers_indoor_office2 IndoorOffice2_dataset.bag replays/
<dataset> is a key of DATASETS (`tiers` and `hr2kilo` are the old names of the first two). --gt is a
TUM file (t x y z qx qy qz qw) for datasets whose ground truth is not in the bag (M3DGR).
Unitree G1 recordings: python make_replay.py g1_robot <bag folder> <out_dir>.
"""
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

POINT = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'),
                  ('ring', '<u2'), ('time', '<f4')])
assert POINT.itemsize == 22
CUSTOM_POINT = np.dtype([('offset_time', '<u4'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                         ('reflectivity', 'u1'), ('tag', 'u1'), ('line', 'u1')])  # ROS 1: packed, 19 B

# reader: 'livox_pc2' = livox_ros_driver2 PointCloud2 (absolute f64 `timestamp`), 'custom' = CustomMsg.
# gt: pose topic in the bag, 'file' = TUM file given with --gt, None = no ground truth.
_TIERS = dict(reader='livox_pc2', lidar='/mid360/livox/lidar', imu='/mid360/livox/imu')
_HR2KILO = dict(reader='custom', lidar='/livox/lidar', imu='/livox/imu', gt=None, gt_kind='return to start',
                platform='Unitree G1 humanoid, Mid-360 in the head')
# M3DGR's mocap poses arrive in bursts (about 3 samples every 10 ms, stamped within microseconds of each
# other: receive time), so only the last sample of each burst is kept (gt_bursts).
_M3DGR = dict(reader='custom', lidar='/livox/mid360/lidar', imu='/livox/mid360/imu', gt='file', gt_bursts=True,
              platform='M3DGR wheeled ground robot (SJTU), Mid-360 tilted forward')
DATASETS = {
    'tiers_indoor_office1': dict(_TIERS, gt='/vrpn_client_node/unitree_b1/pose', gt_kind='motion capture',
                                 platform='Unitree B1 quadruped'),
    'tiers_indoor_office2': dict(_TIERS, gt='/vrpn_client_node/unitree_b1/pose', gt_kind='motion capture',
                                 platform='Unitree B1 quadruped'),
    # /gnss_pose holds latitude [deg], longitude [deg], altitude [m]: converted to local ENU metres.
    'tiers_outdoor_road_cut0': dict(_TIERS, gt='/gnss_pose', gt_kind='GNSS-RTK', gt_geodetic=True,
                                    platform='TIERS outdoor road'),
    'tiers_outdoor_road_cut1': dict(_TIERS, gt='/gnss_pose', gt_kind='GNSS-RTK', gt_geodetic=True,
                                    platform='TIERS outdoor road'),
    'hr2kilo_long_corridor': dict(_HR2KILO),
    'hr2kilo_short_corridor': dict(_HR2KILO),
    'hr2kilo_run': dict(_HR2KILO),
    'hr2kilo_spin': dict(_HR2KILO),
    'm3dgr_dynamic01': dict(_M3DGR, gt_kind='motion capture'),
    'm3dgr_dynamic02': dict(_M3DGR, gt_kind='motion capture'),
    'm3dgr_sha_turn01': dict(_M3DGR, gt_kind='motion capture'),
    # Corridor01's GTCorridor01.txt is one start-to-end transform (ArUco markers), not a trajectory:
    # use it for the end-to-end error.
    'm3dgr_corridor01': dict(_M3DGR, gt=None, gt_kind='start-to-end transform (ArUco), GTCorridor01.txt'),
}
# Unitree G1 (ROS 2 bags of the vendor streams): the vendor cloud is already in the replay layout. The IMU is the
# raw vendor stream, restamped by FAST_LIO (imu_restamp) as on the robot. The replay is named after the bag's
# folder.
_G1 = dict(reader='vendor', lidar='/utlidar/cloud_livox_mid360', gt=None, name_from_bag=True,
           gt_kind='start-to-end alignment',
           platform='Unitree G1 (ours), Mid-360 in the head, vendor stream')
DATASETS['g1_robot'] = dict(_G1, imu='/utlidar/imu_livox_mid360')
# JEPLO (ASIG-Lab, Hugging Face): ROS 2 bags (pass the bag directory), lidar at 20 Hz, Mid-360 upside down;
# the ground truth (Qualisys, gt/<seq>.txt) is the pose of the LiDAR frame, not of the IMU.
_JEPLO = dict(reader='custom', lidar='/livox/lidar', imu='/livox/imu', gt='file',
              gt_kind='motion capture (Qualisys, LiDAR frame)',
              platform='Unitree Go2 quadruped, Mid-360 upside down on the head')
DATASETS.update({
    'jeplo_eil_box': dict(_JEPLO), 'jeplo_eil_mask1': dict(_JEPLO), 'jeplo_eil_mask2': dict(_JEPLO),
    'jeplo_eil_mix': dict(_JEPLO), 'jeplo_eil_stairs': dict(_JEPLO), 'jeplo_long_stairs': dict(_JEPLO),
    'm3dgr_sha_turn02': dict(_M3DGR, gt_kind='motion capture'),
    'm3dgr_corridor02': dict(_M3DGR, gt=None, gt_kind='start-to-end transform (ArUco), GTCorridor02.txt'),
    'm3dgr_elevator01': dict(_M3DGR, gt=None, gt_kind='start-to-end transform (ArUco), GTElevator01.txt'),
})
ALIASES = {'tiers': 'tiers_indoor_office1', 'hr2kilo': 'hr2kilo_long_corridor'}


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def livox_pc2_cloud(reader, conn, raw):
    """livox_ros_driver2 PointCloud2 with absolute per-point `timestamp` (f64 ns)."""
    m = reader.deserialize(raw, conn.msgtype)
    names = [f.name for f in m.fields]
    offs = {f.name: f.offset for f in m.fields}
    src = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'line', 'timestamp'],
                    'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', '<f8'],
                    'offsets': [offs[n] for n in ('x', 'y', 'z', 'intensity', 'line', 'timestamp')],
                    'itemsize': m.point_step})
    assert {'x', 'y', 'z', 'intensity', 'line', 'timestamp'} <= set(names), names
    a = np.frombuffer(m.data.tobytes(), dtype=src, count=m.width * m.height)
    start = stamp_ns(m.header)
    out = np.empty(len(a), dtype=POINT)
    for k in ('x', 'y', 'z', 'intensity'):
        out[k] = a[k]
    out['ring'] = a['line']
    out['time'] = (a['timestamp'] - start).astype(np.float32)
    assert out['time'].min() > -1e3, 'points before the scan stamp'
    return start, out


def vendor_cloud(reader, conn, raw):
    """G1 vendor PointCloud2 (/utlidar/cloud_livox_mid360): already x,y,z,intensity,ring,time (ns)."""
    m = reader.deserialize(raw, conn.msgtype)
    names = [(f.name, f.offset) for f in m.fields]
    assert names == [(n, POINT.fields[n][1]) for n in POINT.names] and m.point_step == POINT.itemsize, names
    return stamp_ns(m.header), np.frombuffer(m.data.tobytes(), dtype=POINT, count=m.width * m.height).copy()


# CustomPoint in ROS 2 CDR: each element starts 4-aligned, so 19 bytes of fields + 1 byte padding.
CUSTOM_POINT_CDR = np.dtype({'names': ['offset_time', 'x', 'y', 'z', 'reflectivity', 'tag', 'line'],
                             'formats': ['<u4', '<f4', '<f4', '<f4', 'u1', 'u1', 'u1'],
                             'offsets': [0, 4, 8, 12, 16, 17, 18], 'itemsize': 20})


def parse_custom_cdr(raw):
    """(header stamp ns, timebase, points) of a ROS 2 (CDR, little endian) CustomMsg."""
    assert raw[:4] == b'\x00\x01\x00\x00', 'not little-endian CDR'
    b = memoryview(raw)[4:]                        # CDR alignment counts from here
    sec, nsec, slen = struct.unpack_from('<iII', b, 0)
    o = 12 + slen                                  # frame_id: length incl. the NUL, then the bytes
    o = (o + 7) & ~7                               # uint64 timebase
    timebase, point_num = struct.unpack_from('<QI', b, o)
    o += 8 + 4 + 1 + 3                             # point_num, lidar_id, rsvd[3]
    o = (o + 3) & ~3
    n, = struct.unpack_from('<I', b, o)
    o += 4
    body = bytes(b[o:o + 20 * n]).ljust(20 * n, b'\0')  # the last element has no trailing padding
    return sec * 1_000_000_000 + nsec, timebase, np.frombuffer(body, dtype=CUSTOM_POINT_CDR, count=n)


def custom_cloud(reader, conn, raw):
    """livox_ros_driver(2) CustomMsg, parsed from the raw bytes (fast for 20 k points): ROS 1 or ROS 2 (CDR).

    offset_time is relative to `timebase`; FAST_LIO takes header.stamp as the scan start, so the two
    must agree (largest difference reported by main)."""
    if getattr(reader, 'is2', False):
        start, timebase, a = parse_custom_cdr(raw)
        n = len(a)
    else:
        seq, sec, nsec, flen = struct.unpack_from('<IIII', raw, 0)
        o = 16 + flen
        timebase, point_num = struct.unpack_from('<QI', raw, o)
        o += 8 + 4 + 1 + 3
        n, = struct.unpack_from('<I', raw, o)
        o += 4
        a = np.frombuffer(raw, dtype=CUSTOM_POINT, count=n, offset=o)
        start = sec * 1_000_000_000 + nsec           # FAST_LIO uses header.stamp as scan start
    custom_cloud.max_offset = max(getattr(custom_cloud, 'max_offset', 0), abs(start - timebase))
    out = np.empty(n, dtype=POINT)
    for k in ('x', 'y', 'z'):
        out[k] = a[k]
    out['intensity'] = a['reflectivity']
    out['ring'] = a['line']
    out['time'] = a['offset_time'].astype(np.float32)  # relative to timebase == header stamp
    return start, out


def gt_pose(m):
    """(position, orientation) of a PoseStamped, PoseWithCovarianceStamped, Odometry or TransformStamped."""
    if hasattr(m, 'transform'):
        return m.transform.translation, m.transform.rotation
    pose = m.pose.pose if hasattr(m.pose, 'pose') else m.pose
    return pose.position, pose.orientation


def geodetic_to_enu(rows):
    """(t, lat, lon, alt, q...) rows -> (t, east, north, up, q...) in metres, WGS-84, origin at the first row."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    g = np.array(rows, dtype=np.float64)
    lat, lon, h = np.radians(g[:, 1]), np.radians(g[:, 2]), g[:, 3]
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    ecef = np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - e2) + h) * np.sin(lat)], axis=1)
    la, lo = lat[0], lon[0]
    R = np.array([[-np.sin(lo), np.cos(lo), 0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    g[:, 1:4] = (ecef - ecef[0]) @ R.T
    return [tuple(r) for r in g]


def load_gt_file(path):
    """TUM rows (t x y z qx qy qz qw); comment and header lines are skipped."""
    rows = []
    for line in open(path):
        f = line.replace(',', ' ').split()
        if len(f) >= 8 and not line.lstrip().startswith('#'):
            try:
                rows.append(tuple(float(v) for v in f[:8]))
            except ValueError:
                pass
    return rows


def main():
    args = sys.argv[1:]
    gt_file = None
    if '--gt' in args:
        i = args.index('--gt')
        gt_file = Path(args[i + 1])
        del args[i:i + 2]
    name, bag, out_dir = ALIASES.get(args[0], args[0]), Path(args[1]), Path(args[2])
    cfg = DATASETS[name]
    if cfg.get('name_from_bag'):   # a folder holding the bag, or its bag/ inside
        name = (bag.parent.name if bag.name == 'bag' else bag.name)
        bag = bag / 'bag' if (bag / 'bag').is_dir() else bag
    gt_topic = cfg['gt'] if cfg['gt'] not in (None, 'file') else None
    if cfg['gt'] == 'file' and gt_file is None:
        sys.exit('%s: the ground truth is a separate file, pass --gt <file>' % name)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, gt = [], []
    n_points = []
    acc_first = []
    # ROS 2 Foxy bags carry no message definitions: give the reader the Foxy type store.
    store = dict(default_typestore=get_typestore(Stores.ROS2_FOXY)) if bag.is_dir() else {}
    readers = dict(livox_pc2=livox_pc2_cloud, custom=custom_cloud, vendor=vendor_cloud)
    with AnyReader([bag], **store) as reader:
        topics = {c.topic for c in reader.connections}
        missing = [t for t in (cfg['lidar'], cfg['imu'], gt_topic) if t and t not in topics]
        if missing:
            sys.exit('%s: topics %s not in the bag; it has %s' % (bag.name, missing, sorted(topics)))
        conns = [c for c in reader.connections if c.topic in (cfg['lidar'], cfg['imu'], gt_topic)]
        for conn, t, raw in reader.messages(connections=conns):
            if conn.topic == cfg['imu']:
                m = reader.deserialize(raw, conn.msgtype)
                a, g = m.linear_acceleration, m.angular_velocity
                records.append((stamp_ns(m.header), 1, struct.pack('<6d', a.x, a.y, a.z, g.x, g.y, g.z)))
                if len(acc_first) < 200:
                    acc_first.append((a.x, a.y, a.z))
            elif conn.topic == cfg['lidar']:
                start, pts = readers[cfg['reader']](reader, conn, raw)
                end = start + int(round(float(pts['time'].max())))
                body = zlib.compress(pts.tobytes(), 1)
                records.append((start, 2, struct.pack('<qII', end, len(pts), len(body)) + body))
                n_points.append(len(pts))
            else:
                m = reader.deserialize(raw, conn.msgtype)
                p, q = gt_pose(m)
                gt.append((stamp_ns(m.header) / 1e9, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    if gt_file is not None:
        gt = load_gt_file(gt_file)
        if cfg.get('gt_bursts'):
            gt = sorted(gt)
            gt = [r for r, nxt in zip(gt, gt[1:] + [None]) if nxt is None or nxt[0] - r[0] > 1e-3]
    if gt and cfg.get('gt_geodetic'):
        gt = geodetic_to_enu(sorted(gt))
    if getattr(custom_cloud, 'max_offset', 0) > 1_000_000:
        print('WARNING: CustomMsg header.stamp differs from timebase by up to %.1f ms' % (custom_cloud.max_offset / 1e6))
    # Mount from the IMU at the start (accel in g): where gravity points in the sensor frame.
    g0 = np.mean(np.array(acc_first), axis=0)
    g0 = g0 / np.linalg.norm(g0)
    tilt = float(np.degrees(np.arccos(np.clip(g0[2], -1.0, 1.0))))   # angle between sensor z and up
    mount = "upright" if tilt < 10 else "upside-down" if tilt > 170 else "tilted %.0f deg" % tilt
    records.sort(key=lambda r: (r[0], r[1]))
    imu_t = np.array([r[0] for r in records if r[1] == 1], dtype=np.int64)
    scans = [r for r in records if r[1] == 2]
    meta = dict(name=name, source=str(bag.name), platform=cfg['platform'], mount=mount,
                gravity_dir_at_start=[round(float(v), 3) for v in g0], tilt_deg=round(tilt, 1),
                ground_truth=cfg.get('gt_kind', 'none') if (gt or cfg['gt'] is None) else 'none',
                lidar_topic_in_bag=cfg['lidar'], imu_topic_in_bag=cfg['imu'],
                n_scans=len(scans), n_imu=len(imu_t),
                duration_s=(records[-1][0] - records[0][0]) / 1e9,
                imu_rate_hz=(len(imu_t) - 1) / ((imu_t[-1] - imu_t[0]) / 1e9),
                points_per_scan=[int(min(n_points)), int(np.median(n_points)), int(max(n_points))],
                point_layout='x,y,z,intensity f32 | ring u16 | time f32 ns (vendor /utlidar layout)',
                imu_acc_unit='g (Livox)', frame_id='livox_frame')
    path = out_dir / (name + '.g1replay')
    with open(path, 'wb') as f:
        mj = json.dumps(meta).encode()
        f.write(b'G1RP' + struct.pack('<II', 1, len(mj)) + mj)
        for stamp, typ, payload in records:
            f.write(struct.pack('<Bq', typ, stamp) + payload)
    print('%s: %d scans, %d IMU (%.1f Hz), %.1f s, %.0f MB' % (
        path.name, len(scans), len(imu_t), meta['imu_rate_hz'], meta['duration_s'], path.stat().st_size / 1e6))
    if gt:
        gt_path = out_dir / (name + '.gt.tum')
        np.savetxt(gt_path, np.array(sorted(gt)), fmt='%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f',
                   header='timestamp x y z qx qy qz qw  (%s, from %s)' % (cfg.get('gt_kind'), gt_file.name if gt_file else gt_topic))
        print('%s: %d poses' % (gt_path.name, len(gt)))
    (out_dir / (name + '.json')).write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
