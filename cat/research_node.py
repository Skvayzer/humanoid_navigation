#!/usr/bin/env python3
"""Optional CAT fields + read-only telemetry. No inference or actuation.

All output topics are visualization-only /g1_cat names. No robot SDK,
controller imports, service/action clients, global TF or motor publishers.
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import JointState, PointCloud2, PointField
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from unitree_hg.msg import LowState

from preview_node import CatPreview
from perception_core import SHAPE, RESOLUTION
from field_core import compute_fields
from state_core import StateBuffer, Kinematics, JOINT_NAMES, SITE_NAMES

PELVIS_FRAME = 'g1_cat/pelvis_preview'


def validate_goal(msg, floor_z):
    if msg.header.frame_id != 'map':
        raise ValueError('CAT preview goal must use map frame')
    p, q = msg.pose.position, msg.pose.orientation
    values = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
    if not np.isfinite(values).all() or not .9 < np.linalg.norm(values[3:]) < 1.1:
        raise ValueError('invalid CAT preview goal pose')
    # 2D click specifies XY, not root height/yaw; explicitly keep upstream
    # navigation field slice at floor + .75 m. This never becomes a robot goal.
    return np.array([p.x, p.y, floor_z+.75])


class ResearchPreview(CatPreview):
    def __init__(self):
        super().__init__()
        self.state_buffer = StateBuffer()
        self.kinematics = Kinematics()
        self.state_error = None
        self.field_lock = threading.RLock()
        self.field_worker = ThreadPoolExecutor(max_workers=1)
        self.field_job = None
        self.field_epoch = 0
        self.field_received = None
        self.field_last_submit = -float('inf')
        self.goal = self.goal_received = None
        self.state_sub = self.create_subscription(
            LowState, '/lowstate', self.receive_state,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self.goal_sub = self.create_subscription(
            PoseStamped, '/g1_cat/goal_preview', self.receive_goal, 1)
        self.joints_pub = self.create_publisher(JointState, '/g1_cat/joint_states_preview', 1)
        self.body_pub = self.create_publisher(Marker, '/g1_cat/body_sites_preview', 1)
        self.gravity_pub = self.create_publisher(Marker, '/g1_cat/pelvis_gravity_preview', 1)
        self.state_pub = self.create_publisher(String, '/g1_cat/robot_state_diagnostics', 1)
        self.sdf_pub = self.create_publisher(PointCloud2, '/g1_cat/sdf_slice', 1)
        self.boundary_pub = self.create_publisher(Marker, '/g1_cat/boundary_preview', 1)
        self.guidance_pub = self.create_publisher(Marker, '/g1_cat/guidance_preview', 1)
        self.field_pub = self.create_publisher(String, '/g1_cat/field_diagnostics', 1)
        self.state_timer = self.create_timer(.1, self.publish_state)
        self.health_timer = self.create_timer(.2, self.check_field_age)
        print('CAT RESEARCH PREVIEW: fields and read-only LowState; policy_ready=false; '
              'pelvis/map calibration UNVERIFIED. No actuator output.', flush=True)

    def status(self, publisher, **details):
        msg = String()
        msg.data = json.dumps(dict(motion_enabled=False, policy_ready=False,
                                   inference_enabled=False, **details), allow_nan=False)
        publisher.publish(msg)

    def marker(self, namespace, frame='map', kind=Marker.LINE_LIST):
        msg = Marker()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns, msg.id, msg.type, msg.action = namespace, 0, kind, Marker.ADD
        msg.pose.orientation.w = 1.
        msg.scale.x = .012
        msg.color.g, msg.color.a = 1., .9
        msg.lifetime.sec = 1 if frame == PELVIS_FRAME else 3
        return msg

    @staticmethod
    def point(p):
        return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))

    def receive_state(self, message):
        try:
            self.state_buffer.receive(message, time.monotonic())
            self.state_error = None
        except ValueError as exc:
            self.state_error = str(exc)

    def publish_state(self):
        try:
            state = self.state_buffer.snapshot(time.monotonic())
            sites = self.kinematics.sites(state['q'])
            joints = JointState()
            joints.header.stamp = self.get_clock().now().to_msg()
            joints.header.frame_id = PELVIS_FRAME
            joints.name, joints.position = list(JOINT_NAMES), state['q'].tolist()
            joints.velocity, joints.effort = state['dq'].tolist(), state['tau'].tolist()
            self.joints_pub.publish(joints)
            body = self.marker('body_sites', PELVIS_FRAME, Marker.SPHERE_LIST)
            body.scale.x = body.scale.y = body.scale.z = .045
            body.color.b = 1.
            body.points = [self.point(sites[n][:3, 3]) for n in SITE_NAMES]
            self.body_pub.publish(body)
            arrow = self.marker('pelvis_gravity', PELVIS_FRAME, Marker.ARROW)
            arrow.scale.x, arrow.scale.y, arrow.scale.z = .025, .055, .07
            arrow.color.r, arrow.color.g = 1., .5
            start = sites['imu_in_pelvis'][:3, 3]
            arrow.points = [self.point(start), self.point(start+.3*state['gravity'])]
            self.gravity_pub.publish(arrow)
            self.status(self.state_pub, state='TELEMETRY_OK_UNCALIBRATED',
                        tick=state['tick'], receive_age_s=time.monotonic()-self.state_buffer.received,
                        mode_machine=state['mode_machine'],
                        model='upstream_g1_29dof_rev_1_0', embodiment_verified=False,
                        map_alignment_verified=False, frame=PELVIS_FRAME,
                        quaternion_wxyz=state['quaternion_wxyz'].tolist(),
                        gyro=state['gyro'].tolist(), acceleration=state['acceleration'].tolist(),
                        gravity=state['gravity'].tolist(), site_order=list(SITE_NAMES))
        except ValueError as exc:
            for publisher, name in ((self.body_pub, 'body_sites'), (self.gravity_pub, 'pelvis_gravity')):
                marker = self.marker(name, PELVIS_FRAME)
                marker.action = Marker.DELETE
                publisher.publish(marker)
            self.status(self.state_pub, state='WAITING_OR_INVALID', reason=self.state_error or str(exc))

    def receive_goal(self, msg):
        try:
            goal = validate_goal(msg, self.cfg['floor_z'])
        except ValueError as exc:
            with self.field_lock:
                self.goal = self.goal_received = None
                self.clear_fields('invalid goal: '+str(exc))
            return
        with self.field_lock:
            self.goal, self.goal_received = goal, time.monotonic()
            self.clear_fields('new preview goal; waiting for fresh fields')

    def on_sample(self, grid, processed, origin, scan):
        if not hasattr(self, 'field_worker'):
            return
        with self.field_lock:
            now = time.monotonic()
            if now-self.field_last_submit < 2 or (self.field_job is not None and not self.field_job.done()):
                return
            self.field_last_submit = now
            goal = None if self.goal is None else self.goal.copy()
            self.field_job = self.field_worker.submit(self.compute_and_publish,
                grid.copy(), processed.copy(), origin.copy(), scan, goal, self.field_epoch, self.sequence)

    def on_invalid(self, reason):
        if hasattr(self, 'field_lock'):
            with self.field_lock:
                self.clear_fields('perception invalid: '+reason)

    def clear_fields(self, reason):
        self.field_epoch += 1
        self.field_received = None
        for pub, name in ((self.boundary_pub, 'boundary'), (self.guidance_pub, 'guidance')):
            msg = self.marker(name)
            msg.action = Marker.DELETE
            pub.publish(msg)
        self.publish_slice(np.empty((0, 4), np.float32))
        self.status(self.field_pub, state='WAITING_OR_INVALID', reason=reason,
                    unknown_space='blocked', body_sampling_ready=False)

    def check_field_age(self):
        with self.field_lock:
            now = time.monotonic()
            if self.goal_received is not None and now-self.goal_received > 60:
                self.goal = self.goal_received = None
                self.clear_fields('preview goal expired after 60 seconds')
            elif self.field_received is not None and now-self.field_received > self.cfg['max_output_age']:
                self.clear_fields('field input expired; old arrows removed')

    def publish_slice(self, values):
        msg = PointCloud2()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = 1, len(values)
        msg.fields = [PointField(name=n, offset=i*4, datatype=7, count=1)
                      for i, n in enumerate(('x', 'y', 'z', 'intensity'))]
        msg.is_dense = True
        msg.point_step, msg.row_step = 16, 16*len(values)
        msg.data = np.ascontiguousarray(values, dtype='<f4').tobytes()
        self.sdf_pub.publish(msg)

    def compute_and_publish(self, grid, processed, origin, scan, goal, epoch, sequence):
        started = time.monotonic()
        try:
            if started-scan.received > self.cfg['max_output_age']:
                raise ValueError('field input already stale')
            fields = compute_fields(grid, processed, origin, goal)
            z = int(np.clip(np.floor((self.cfg['floor_z']+.75-origin[2])/RESOLUTION), 0, SHAPE[2]-1))
            ij = np.array(list(np.ndindex(32, 32))) * 4
            indices = np.column_stack((ij, np.full(len(ij), z)))
            indices = indices[fields.known_free[tuple(indices.T)]]
            positions = origin+(indices+.5)*RESOLUTION
            distances = fields.sdf[tuple(indices.T)]
            markers = []
            for name, array, color in (('boundary', fields.boundary, (1., .5, 0.)),
                                        ('guidance', fields.guidance, (0., 1., 1.))):
                marker = self.marker(name)
                marker.color.r, marker.color.g, marker.color.b = color
                vectors = array[tuple(indices.T)]
                for p, v in zip(positions, vectors):
                    norm = np.linalg.norm(v)
                    if norm > 1e-6:
                        end = p+.12*v/norm
                        # Shaft plus two arrowhead segments; bounded <1024 arrows.
                        tangent = np.cross(v/norm, [0., 0., 1.])
                        if np.linalg.norm(tangent) < 1e-6:
                            tangent = np.cross(v/norm, [0., 1., 0.])
                        tangent /= np.linalg.norm(tangent)
                        for a, b in ((p, end), (end, end-.035*v/norm+.02*tangent),
                                     (end, end-.035*v/norm-.02*tangent)):
                            marker.points.extend([self.point(a), self.point(b)])
                markers.append(marker)
            with self.field_lock:
                if epoch != self.field_epoch:
                    return  # an invalidation or a different goal superseded this work
                if time.monotonic()-scan.received > self.cfg['max_output_age']:
                    raise ValueError('field calculation exceeded freshness deadline')
                self.publish_slice(np.column_stack((positions, distances)))
                self.boundary_pub.publish(markers[0])
                self.guidance_pub.publish(markers[1])
                self.field_received = scan.received
                self.status(self.field_pub, state='FIELDS_PREVIEW_OK' if goal is not None else 'DISTANCE_ONLY_NO_GOAL',
                            sequence=sequence, source_stamp_ns=scan.end_ns,
                            output_receive_age_s=time.monotonic()-scan.received,
                            processing_ms=(time.monotonic()-started)*1000,
                            origin=origin.tolist(), slice_z=float(origin[2]+(z+.5)*RESOLUTION),
                            known_free=int(fields.known_free.sum()), reachable=int(fields.reachable.sum()),
                            goal=None if goal is None else goal.tolist(), unknown_space='blocked',
                            body_sampling_ready=False, frame='map')
        except Exception as exc:
            with self.field_lock:
                if epoch == self.field_epoch:
                    self.clear_fields(type(exc).__name__+': '+str(exc))


def main():
    rclpy.init()
    node = ResearchPreview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.timer.cancel()
        node.state_timer.cancel()
        node.health_timer.cancel()
        node.worker.shutdown(wait=True)
        node.field_worker.shutdown(wait=True)
        if rclpy.ok():
            node.invalidate('research preview stopped', hard=True)
        node.tree.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
