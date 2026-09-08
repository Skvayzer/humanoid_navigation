#!/usr/bin/env python3
"""Perception-only CAT preview. Publishes visualization under /g1_cat ONLY."""
import json
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener

from core import (OccupancyMap, SHAPE, RESOLUTION, cat_preprocess, centers,
                  cloud_xyz, grid_origin, rotation_matrix, transform_points)

CLOUD_TOPICS = ("input_cloud", "ground_points", "obstacle_points", "occupancy_raw", "occupancy_cat")


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class CatPreview(Node):
    def __init__(self):
        # Disable implicit parameter service and rosout publishers as well.
        super().__init__("cat_perception_preview", namespace="g1_cat",
                         enable_rosout=False, start_parameter_services=False)
        defaults = {"input_topic": "/g1_slam/cloud_registered_body", "fixed_frame": "map",
                    "source_frame": "body", "floor_z": 0.0, "ground_cutoff": 0.10,
                    "rate_hz": 1.0, "max_points": 20000, "max_range": 2.5,
                    "max_receive_age": 0.5, "max_tf_delta": 0.25,
                    "map_reset_seconds": 10.0, "max_octomap_nodes": 1000000,
                    "sensor_origin_body": [-0.011, -0.02329, 0.04412],
                    "snapshot_directory": "/data", "snapshot_every": 5,
                    "arm_token": "/run/g1_nav/g1_motion_armed"}
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.cfg = {key: self.get_parameter(key).value for key in defaults}
        self.validate_config()
        self.tree = OccupancyMap()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(PointCloud2, self.cfg["input_topic"],
                                                       self.receive, qos)
        # Reliable publishers also serve best-effort readers; tiny queue, 1 Hz.
        self.cloud_pubs = {name: self.create_publisher(PointCloud2, "/g1_cat/" + name, 1)
                           for name in CLOUD_TOPICS}
        self.roi_pub = self.create_publisher(Marker, "/g1_cat/roi", 1)
        self.status_pub = self.create_publisher(String, "/g1_cat/diagnostics", 1)
        self.latest = None
        self.last_stamp = None
        self.last_origin = None
        self.last_pose = None
        self.reset_time = time.monotonic()
        self.sequence = 0
        self.last_status = None
        self.timer = self.create_timer(1.0 / self.cfg["rate_hz"], self.tick)
        print("CAT PERCEPTION ONLY: no SDK, policy, shared memory, TF or motion publishers.", flush=True)

    def validate_config(self):
        c = self.cfg
        # This adapter's calibrated origin is for FAST-LIO's existing body cloud.
        if c["input_topic"] != "/g1_slam/cloud_registered_body" or c["source_frame"] != "body" or c["fixed_frame"] != "map":
            raise ValueError("this preview requires the existing body cloud and floor-aligned map")
        bounds = {"rate_hz": (0.1, 2.0), "max_points": (100, 50000),
                  "max_range": (0.2, 2.5), "ground_cutoff": (0.0, 0.3),
                  "max_receive_age": (0.05, 1.0), "max_tf_delta": (0.01, 0.5),
                  "map_reset_seconds": (1.0, 30.0), "max_octomap_nodes": (1000, 1000000),
                  "snapshot_every": (1, 60)}
        for key, (low, high) in bounds.items():
            if not np.isfinite(c[key]) or not low <= c[key] <= high:
                raise ValueError("invalid " + key)
        if not np.isfinite(c["floor_z"]) or abs(c["floor_z"]) > 10:
            raise ValueError("invalid floor_z")
        if abs(c["floor_z"]/RESOLUTION - round(c["floor_z"]/RESOLUTION)) > 1e-6:
            raise ValueError("floor_z must align to the 4 cm OctoMap lattice")
        origin = np.asarray(c["sensor_origin_body"])
        if origin.shape != (3,) or not np.isfinite(origin).all() or np.linalg.norm(origin) > 0.2:
            raise ValueError("invalid LiDAR origin in IMU/body")

    def receive(self, cloud):
        self.latest = (cloud, time.monotonic())

    def publish_cloud(self, name, points, stamp):
        msg = PointCloud2()
        msg.header.frame_id = "map"
        msg.header.stamp = stamp
        msg.height, msg.width = 1, len(points)
        msg.fields = [PointField(name=n, offset=i*4, datatype=7, count=1)
                      for i, n in enumerate(("x", "y", "z"))]
        msg.is_bigendian, msg.is_dense = False, True
        msg.point_step, msg.row_step = 12, len(points)*12
        msg.data = np.ascontiguousarray(points, dtype="<f4").tobytes()
        self.cloud_pubs[name].publish(msg)

    def report(self, state, **details):
        # Preview validity is NOT a permission to move or proof of localization.
        data = dict(state=state, perception_only=True, motion_enabled=False,
                    policy_ready=False, sequence=self.sequence,
                    frame="map", shape=list(SHAPE), resolution=RESOLUTION, **details)
        message = String()
        message.data = json.dumps(data, sort_keys=True, allow_nan=False)
        self.status_pub.publish(message)
        if state != self.last_status or self.sequence % 5 == 0:
            print(message.data, flush=True)
        self.last_status = state

    def invalidate(self, reason):
        self.tree.reset()
        self.last_origin = self.last_pose = None
        self.reset_time = time.monotonic()
        stamp = self.get_clock().now().to_msg()
        for name in CLOUD_TOPICS:
            self.publish_cloud(name, np.empty((0, 3)), stamp)
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "cat_roi", 0
        marker.action = Marker.DELETE
        self.roi_pub.publish(marker)
        self.report("WAITING_OR_INVALID", reason=reason)

    def lookup(self, cloud):
        # NDT/FAST-LIO TF uses sensor time, not necessarily wall time. First try
        # the scan stamp. If NDT lags, allow the latest common time ONLY within
        # the configured bound; never substitute an identity transform.
        stamp = Time.from_msg(cloud.header.stamp)
        mode = "scan_time"
        try:
            result = self.tf_buffer.lookup_transform("map", "body", stamp)
        except Exception:
            result = self.tf_buffer.lookup_transform("map", "body", Time())
            mode = "latest_bounded"
        delta = stamp_seconds(cloud.header.stamp) - stamp_seconds(result.header.stamp)
        if abs(delta) > self.cfg["max_tf_delta"]:
            raise ValueError("map/body TF is stale relative to scan (%.3f s)" % delta)
        t, q = result.transform.translation, result.transform.rotation
        return [t.x, t.y, t.z], [q.x, q.y, q.z, q.w], mode, delta

    def tick(self):
        started = time.monotonic()
        try:
            if Path(self.cfg["arm_token"]).exists():
                self.invalidate("motion arm token exists; preview paused (does NOT disarm robot)")
                return
            if self.latest is None:
                self.invalidate("waiting for " + self.cfg["input_topic"])
                return
            cloud, received = self.latest
            age = started - received
            stamp = stamp_seconds(cloud.header.stamp)
            if age > self.cfg["max_receive_age"] or stamp <= 0:
                raise ValueError("stale/invalid cloud timestamp or no fresh receive")
            if self.last_stamp is not None and stamp <= self.last_stamp:
                if stamp < self.last_stamp:
                    self.last_stamp = stamp  # recovery requires a NEW advancing scan
                raise ValueError("cloud timestamp stopped or went backwards")
            if cloud.header.frame_id != self.cfg["source_frame"]:
                raise ValueError("unexpected source frame: " + cloud.header.frame_id)
            translation, quat, tf_mode, tf_delta = self.lookup(cloud)
            xyz = cloud_xyz(cloud, self.cfg["max_points"])
            if len(xyz) < 10:
                raise ValueError("too few finite input points")
            points = transform_points(xyz, translation, quat)
            sensor = transform_points([self.cfg["sensor_origin_body"]], translation, quat)[0]
            origin = grid_origin(translation, self.cfg["floor_z"])
            reset_reason = "none"
            if started - self.reset_time > self.cfg["map_reset_seconds"]:
                reset_reason = "bounded_history_expired"
            if self.last_origin is not None and np.linalg.norm(origin[:2]-self.last_origin[:2]) > 1.0:
                reset_reason = "local_window_moved"
            if self.last_pose is not None:
                old_t, old_r = self.last_pose
                angle = np.arccos(np.clip((np.trace(old_r.T @ rotation_matrix(quat))-1)/2, -1, 1))
                if np.linalg.norm(np.asarray(translation)-old_t) > 0.5 or angle > np.deg2rad(30):
                    reset_reason = "large_pose_change"
            if reset_reason != "none":
                self.tree.reset()
                self.reset_time = started
                self.last_origin = None
            if self.last_origin is None:
                self.last_origin = origin.copy()
            self.last_pose = (np.asarray(translation), rotation_matrix(quat))
            heights = points[:, 2] - self.cfg["floor_z"]
            ground = heights < self.cfg["ground_cutoff"]
            self.tree.insert(points, ~ground, sensor, self.cfg["max_range"])
            if self.tree.nodes > self.cfg["max_octomap_nodes"]:
                raise ValueError("OctoMap node budget exceeded; history cleared")
            grid = self.tree.export(origin)
            raw = grid == 2
            processed = cat_preprocess(raw)
            within = (np.linalg.norm(points-sensor, axis=1) <= self.cfg["max_range"])
            within &= (points >= origin).all(axis=1) & (points < origin + np.array(SHAPE)*RESOLUTION).all(axis=1)
            now_msg = self.get_clock().now().to_msg()
            for name, values in (("input_cloud", points[within]), ("ground_points", points[within & ground]),
                                 ("obstacle_points", points[within & ~ground]),
                                 ("occupancy_raw", centers(raw, origin)),
                                 ("occupancy_cat", centers(processed, origin))):
                self.publish_cloud(name, values, now_msg)
            self.publish_roi(origin, now_msg)
            self.last_stamp = stamp
            self.sequence += 1
            details = dict(origin=origin.tolist(), sensor_origin=sensor.tolist(),
                           input_points=cloud.width*cloud.height, sampled_points=len(xyz),
                           obstacle_returns=int((within & ~ground).sum()),
                           ground_returns=int((within & ground).sum()),
                           occupied_raw=int(raw.sum()), occupied_cat=int(processed.sum()),
                           cat_added=int((processed & ~raw).sum()), cat_removed=int((raw & ~processed).sum()),
                           known_free=int((grid == 1).sum()), unknown=int((grid == 0).sum()),
                           source_stamp=stamp, receive_age_s=age, tf_mode=tf_mode, tf_delta_s=tf_delta,
                           wall_minus_sensor_s=self.get_clock().now().nanoseconds*1e-9-stamp,
                           octomap_nodes=self.tree.nodes, history_age_s=started-self.reset_time,
                           history_reset=reset_reason)
            if self.cfg["snapshot_directory"] and self.sequence % self.cfg["snapshot_every"] == 0:
                directory = Path(self.cfg["snapshot_directory"])
                directory.mkdir(parents=True, exist_ok=True)
                # One bounded latest snapshot, never a policy SHM buffer.
                target = directory / "latest.npz"
                temporary = directory / "latest.tmp.npz"
                np.savez_compressed(str(temporary), occupancy_state=grid,
                                    cat_occupancy=processed.astype(np.uint8), origin=origin,
                                    resolution=RESOLUTION, source_stamp=stamp,
                                    metadata=json.dumps(details))
                os.replace(str(temporary), str(target))
            details["processing_ms"] = (time.monotonic()-started)*1000
            if time.monotonic() - received > 1.5:
                raise ValueError("processing output too old; decrease max_points")
            self.report("PREVIEW_OK", **details)
        except Exception as exc:
            self.invalidate(type(exc).__name__ + ": " + str(exc))

    def publish_roi(self, origin, stamp):
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = "map", stamp
        marker.ns, marker.id, marker.type, marker.action = "cat_roi", 0, Marker.LINE_LIST, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.015
        marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.8
        marker.lifetime.sec = 3
        size = np.array(SHAPE)*RESOLUTION
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            for a in (0, 1):
                for b in (0, 1):
                    start = np.array(origin)
                    start[others] += size[others] * [a, b]
                    end = start.copy()
                    end[axis] += size[axis]
                    for p in (start, end):
                        marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        self.roi_pub.publish(marker)


def main():
    rclpy.init()
    node = CatPreview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.invalidate("preview stopped")
        node.tree.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
