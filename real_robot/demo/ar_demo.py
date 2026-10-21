#!/usr/bin/env python3
"""Demo recorder: the head camera with the virtual world drawn in (AR), and a bag of how the robot sees the world.

  ar_demo.py --out /data/demo [--fps 10] [--stop-after 4]

* Camera: JPEG frames from Unitree's videohub (the SDK's VideoClient RPC, api 1001, over /api/videohub/request and
  /api/videohub/response): no camera device is opened, the vendor video service keeps running.
* Overlay: the executor's /g1_wb/virtual_scene (boxes, route, waypoints, geofence) and /g1_wb/goal, projected with the
  D435i colour intrinsics and T_map_cam = T_map_body (SLAM) . T_body_torso (LiDAR mount) . T_torso_d435 (URDF) . optical.
* Always publishes /g1_wb/ar/compressed (Foxglove). From the moment the executor walks (status: POLICY, not holding)
  until it has stood still / left POLICY for --stop-after s, writes <out>/<time>/ar.mp4 and <out>/<time>/bag (rosbag2:
  the viewer topics + the AR image), to be played side by side in Foxglove for a presentation.
Read-only towards the robot: it publishes nothing but the videohub request and the AR image.
"""
import argparse
import array
import json
import math
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# D435i colour camera, factory calibration read from the device (pyrealsense2), per resolution: fx, fy, cx, cy
INTRINSICS = {(640, 480): (605.8, 606.2, 321.2, 253.2), (1280, 720): (908.7, 909.2, 641.8, 379.8), (424, 240): (302.9, 303.1, 212.6, 126.6)}
# torso_link -> Mid-360 (real_robot/wholebody frames.MOUNTS['unitree_urdf']) and the FAST-LIO IMU offset (frames.LIDAR_TO_IMU_T)
MOUNT_XYZ, MOUNT_RPY = (0.0002835, 0.00003, 0.428434), (math.pi, 0.05112069379091391, 0.0)
LIDAR_TO_IMU_T = (-0.011, -0.02329, 0.04412)
# torso_link -> d435_link (g1_29dof_rev_1_0.urdf d435_joint); the colour imager sits 15 mm to the right of it
D435_XYZ, D435_RPY = (0.0576235, 0.01753, 0.42987), (0.0, 0.8307767239493009, 0.0)
COLOR_IN_LINK = (0.0, -0.0152, 0.0)
VIEWER_TOPICS = {   # bagged while walking (+ the AR image)
    '/g1_slam/map': 'sensor_msgs/msg/PointCloud2', '/g1_slam/obstacles': 'sensor_msgs/msg/PointCloud2',
    '/g1_wb/virtual_scene': 'visualization_msgs/msg/MarkerArray', '/g1_wb/sdf_slice': 'sensor_msgs/msg/PointCloud2',
    '/g1_wb/sites': 'sensor_msgs/msg/PointCloud2', '/g1_wb/pelvis': 'geometry_msgs/msg/PoseStamped',
    '/g1_wb/goal': 'geometry_msgs/msg/PoseStamped', '/g1_wb/status': 'std_msgs/msg/String', '/tf': 'tf2_msgs/msg/TFMessage',
    '/tf_static': 'tf2_msgs/msg/TFMessage', '/g1_slam/odometry': 'nav_msgs/msg/Odometry'}
VIDEOHUB_GET_IMAGE = 1001


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def tf(r, t):
    m = np.eye(4); m[:3, :3] = r; m[:3, 3] = t
    return m


def quat_m(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


T_TORSO_BODY = tf(rpy(*MOUNT_RPY), MOUNT_XYZ) @ tf(np.eye(3), -np.asarray(LIDAR_TO_IMU_T))
T_BODY_TORSO = np.linalg.inv(T_TORSO_BODY)
OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])     # optical x right, y down, z forward, in link axes
T_TORSO_CAM = tf(rpy(*D435_RPY), D435_XYZ) @ tf(OPTICAL, COLOR_IN_LINK)


class Projector:
    def __init__(self, t_map_cam, size):
        w, h = size
        fx, fy, cx, cy = INTRINSICS.get((w, h)) or tuple(v * w / 640 for v in INTRINSICS[(640, 480)][:3]) + (INTRINSICS[(640, 480)][3] * h / 480,)
        self.k = (fx, fy, cx, cy); self.t_cam_map = np.linalg.inv(t_map_cam)

    def cam(self, pts):
        p = np.asarray(pts, float).reshape(-1, 3)
        return p @ self.t_cam_map[:3, :3].T + self.t_cam_map[:3, 3]

    def pix(self, pc):
        fx, fy, cx, cy = self.k
        return np.column_stack([fx * pc[:, 0] / pc[:, 2] + cx, fy * pc[:, 1] / pc[:, 2] + cy])

    def segment(self, a, b, near=0.05):
        """Pixel ends of a 3-D segment clipped to the camera's front, or None."""
        pa, pb = self.cam([a, b])
        if pa[2] < near and pb[2] < near:
            return None
        if pa[2] < near:
            pa = pa + (pb - pa) * (near - pa[2]) / (pb[2] - pa[2])
        elif pb[2] < near:
            pb = pb + (pa - pb) * (near - pb[2]) / (pa[2] - pb[2])
        q = self.pix(np.array([pa, pb]))
        return tuple(map(int, np.round(q[0]))), tuple(map(int, np.round(q[1])))


def box_corners(m):
    r = quat_m(m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z)
    c = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z]); h = np.array([m.scale.x, m.scale.y, m.scale.z]) / 2
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return c + (signs * h) @ r.T


BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
BOX_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def draw(img, proj, markers, goal, status):
    over = img.copy()
    for m in markers:
        if m.ns == 'virtual_scene':
            corners = box_corners(m); pc = proj.cam(corners)
            for f in BOX_FACES:                                     # translucent faces fully in front of the camera
                if (pc[list(f), 2] > 0.05).all():
                    cv2.fillConvexPoly(over, np.round(proj.pix(pc[list(f)])).astype(np.int32), (26, 115, 255))
            for a, b in BOX_EDGES:
                s = proj.segment(corners[a], corners[b])
                if s:
                    cv2.line(img, s[0], s[1], (0, 90, 255), 2, cv2.LINE_AA)
        elif m.ns in ('path', 'geofence') and len(m.points) > 1:
            color = (80, 220, 60) if m.ns == 'path' else (40, 40, 230)
            pts = [np.array([p.x, p.y, p.z]) for p in m.points]
            for a, b in zip(pts, pts[1:]):
                s = proj.segment(a, b)
                if s:
                    cv2.line(img, s[0], s[1], color, 3 if m.ns == 'path' else 2, cv2.LINE_AA)
    img[:] = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    if goal is not None:
        pc = proj.cam([goal, goal * [1, 1, 0]])
        if (pc[:, 2] > 0.05).all():
            q = np.round(proj.pix(pc)).astype(int)
            cv2.line(img, tuple(q[1]), tuple(q[0]), (80, 220, 60), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(q[1]), 9, (80, 220, 60), 2, cv2.LINE_AA)
    text = '%s  %s  %s  obstacles: %s' % (time.strftime('%H:%M:%S'), status.get('mode', '?'), status.get('scene', ''), status.get('obstacles', '?'))
    cv2.putText(img, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='/data/demo'); ap.add_argument('--fps', type=float, default=10.)
    ap.add_argument('--stop-after', type=float, default=4.0, help='seconds standing / out of POLICY before a take ends')
    ap.add_argument('--always', action='store_true', help='record from start to stop, ignoring the executor state')
    args = ap.parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from rclpy.serialization import serialize_message
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs.msg import CompressedImage
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray
    from geometry_msgs.msg import PoseStamped
    from unitree_api.msg import Request, Response
    import rosbag2_py

    rclpy.init()
    node = Node('ar_demo', namespace='g1_wb', cli_args=['--ros-args', '-r', 'tf:=/tf', '-r', 'tf_static:=/tf_static'])
    log = node.get_logger(); lock = threading.Lock()
    state = dict(jpeg=None, jpeg_id=None, odom=None, markers=[], goal=None, status={}, take=None, quiet_since=None)
    tfb = Buffer(); TransformListener(tfb, node)
    req_pub = node.create_publisher(Request, '/api/videohub/request', 10)
    ar_pub = node.create_publisher(CompressedImage, '/g1_wb/ar/compressed', 2)

    def on_resp(m):
        if m.header.identity.api_id == VIDEOHUB_GET_IMAGE and m.header.status.code == 0 and len(m.binary):
            with lock:
                state['jpeg'] = bytes(m.binary); state['jpeg_id'] = m.header.identity.id
    node.create_subscription(Response, '/api/videohub/response', on_resp, 10)
    node.create_subscription(Odometry, '/g1_slam/odometry', lambda m: state.__setitem__('odom', m), 10)
    node.create_subscription(MarkerArray, '/g1_wb/virtual_scene',
                             lambda m: state.__setitem__('markers', [k for k in m.markers if k.action != 3]), 2)
    node.create_subscription(PoseStamped, '/g1_wb/goal', lambda m: state.__setitem__(
        'goal', np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])), 2)

    def on_status(m):
        try:
            state['status'] = json.loads(m.data)
        except ValueError:
            pass
    node.create_subscription(String, '/g1_wb/status', on_status, 5)

    # raw subscriptions: bag bytes as they come, only while a take is open
    def bagger(topic):
        def cb(raw):
            take = state['take']
            if take is not None:
                with lock:
                    take['writer'].write(topic, raw, node.get_clock().now().nanoseconds)
        return cb
    for topic, typ in VIEWER_TOPICS.items():
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL) if topic == '/tf_static' else 10
        node.create_subscription(get_message(typ), topic, bagger(topic), qos, raw=True)

    def open_take():
        d = Path(args.out) / time.strftime('%Y%m%d_%H%M%S'); d.mkdir(parents=True, exist_ok=True)
        w = rosbag2_py.SequentialWriter()
        w.open(rosbag2_py.StorageOptions(uri=str(d / 'bag'), storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
        for topic, typ in list(VIEWER_TOPICS.items()) + [('/g1_wb/ar/compressed', 'sensor_msgs/msg/CompressedImage')]:
            w.create_topic(rosbag2_py.TopicMetadata(name=topic, type=typ, serialization_format='cdr'))
        log.info('take %s: recording' % d)
        return dict(dir=d, writer=w, video=None, frames=0, started=time.time())

    def close_take(take):
        if take['video'] is not None:
            take['video'].release()
        del take['writer']                                   # closes the bag
        log.info('take %s: %d frames, %.0f s' % (take['dir'], take['frames'], time.time() - take['started']))

    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1)); signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    period = 1.0 / args.fps; next_t = time.monotonic(); rid = int(time.time() * 1000) << 8; last_drawn = None
    log.info('AR demo: videohub frames at %.0f Hz -> /g1_wb/ar/compressed; takes in %s' % (args.fps, args.out))
    try:
        while rclpy.ok() and not stop:      # docker stop: SIGTERM -> the take is closed (mp4 finalised)
            rclpy.spin_once(node, timeout_sec=0.005)
            now = time.monotonic()
            if now < next_t:
                continue
            next_t = now + period
            rid += 1; r = Request(); r.header.identity.id = rid; r.header.identity.api_id = VIDEOHUB_GET_IMAGE
            req_pub.publish(r)
            with lock:
                jpeg, jid = state['jpeg'], state['jpeg_id']
            # takes follow the executor: open while walking, close after --stop-after s of standing / no policy
            st = state['status']; walking = args.always or (st.get('mode') == 'POLICY' and not st.get('hold'))
            if walking:
                state['quiet_since'] = None
                if state['take'] is None:
                    state['take'] = open_take()
            elif state['take'] is not None:
                state['quiet_since'] = state['quiet_since'] or now
                if now - state['quiet_since'] > args.stop_after:
                    take, state['take'] = state['take'], None
                    with lock:
                        close_take(take)
            if jpeg is None or jid == last_drawn:
                continue
            last_drawn = jid
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            odom = state['odom']
            if odom is not None:
                try:
                    t = tfb.lookup_transform('map', odom.header.frame_id or 'camera_init', Time()).transform
                    t_map_odom = tf(quat_m(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z), [t.translation.x, t.translation.y, t.translation.z])
                    p, o = odom.pose.pose.position, odom.pose.pose.orientation
                    t_map_cam = t_map_odom @ tf(quat_m(o.w, o.x, o.y, o.z), [p.x, p.y, p.z]) @ T_BODY_TORSO @ T_TORSO_CAM
                    draw(img, Projector(t_map_cam, (img.shape[1], img.shape[0])), state['markers'], state['goal'], st)
                except Exception as exc:                     # no map frame yet: plain camera
                    log.warn('overlay: %s' % exc, throttle_duration_sec=10.)
            ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            msg = CompressedImage(); msg.header.stamp = node.get_clock().now().to_msg(); msg.header.frame_id = 'd435_color_optical'
            msg.format = 'jpeg'; msg.data = array.array('B', enc.tobytes())    # Foxy: bytes into uint8[] is slow
            ar_pub.publish(msg)
            take = state['take']
            if take is not None:
                with lock:
                    take['writer'].write('/g1_wb/ar/compressed', serialize_message(msg), node.get_clock().now().nanoseconds)
                if take['video'] is None:
                    take['video'] = cv2.VideoWriter(str(take['dir'] / 'ar.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                                                    (img.shape[1], img.shape[0]))
                take['video'].write(img); take['frames'] += 1
    except KeyboardInterrupt:
        pass
    finally:
        if state['take'] is not None:
            close_take(state['take'])
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
