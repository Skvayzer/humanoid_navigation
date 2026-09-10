#!/usr/bin/env python3
"""Passive CAT perception: raw timed LiDAR, existing SLAM poses, no motion control."""
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from livox_ros_driver2.msg import CustomMsg
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener

from perception_core import (OccupancyMap, SHAPE, RESOLUTION, Scan, ScanBuffer,
                             cat_preprocess, centers, deskew_livox, grid_origin,
                             livox_points, rotation_matrix, transform_points, decode_livox_cdr)

CLOUD_TOPICS = ("input_cloud", "ground_points", "obstacle_points", "nearfield_points",
                "occupancy_raw", "occupancy_cat")


def stamp_ns(stamp):
    return stamp.sec * 1000000000 + stamp.nanosec


class CatPreview(Node):
    def __init__(self):
        # Foxy's listener uses relative names. Only its INPUTS are remapped.
        super().__init__("cat_perception_preview", namespace="g1_cat",
                         cli_args=["--ros-args", "-r", "tf:=/tf", "-r", "tf_static:=/tf_static"],
                         enable_rosout=False, start_parameter_services=False)
        defaults = {"input_topic": "/livox/lidar", "fixed_frame": "map",
                    "source_frame": "livox_frame", "floor_z": 0.0, "ground_cutoff": 0.10,
                    "rate_hz": 1.0, "max_points": 20000, "max_range": 2.5,
                    "min_range": 0.0, "max_receive_age": 1.5, "max_output_age": 2.0,
                    "display_hold_seconds": 3.0, "cell_ttl_seconds": 30.0,
                    "max_octomap_nodes": 1000000,
                    "sensor_origin_body": [-0.011, -0.02329, 0.04412],
                    "snapshot_directory": "/data", "snapshot_every": 5}
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.cfg = {key: self.get_parameter(key).value for key in defaults}
        self.validate_config()
        self.tree = OccupancyMap()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=2, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            CustomMsg, self.cfg["input_topic"], self.receive, qos, raw=True)
        self.cloud_pubs = {name: self.create_publisher(PointCloud2, "/g1_cat/" + name, 1)
                           for name in CLOUD_TOPICS}
        self.roi_pub = self.create_publisher(Marker, "/g1_cat/roi", 1)
        self.status_pub = self.create_publisher(String, "/g1_cat/diagnostics", 1)
        self.scans = ScanBuffer()
        self.last_stamp_ns = -1
        self.generation = 0
        self.last_pose = None
        self.last_origin = None
        self.last_good_received = None
        self.reset_time = time.monotonic()
        self.input_error = None
        self.sequence = 0
        self.last_status = None
        self.last_report_time = 0.0
        # One bounded processing task, independent of the ROS receive loop.
        # Avoid Foxy's Python MultiThreadedExecutor ready-callback contention.
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.timer = self.create_timer(1.0 / self.cfg["rate_hz"], self.schedule)
        print("CAT PERCEPTION ONLY: raw LiDAR, no SDK, policy, TF or motion publishers.", flush=True)

    def validate_config(self):
        c = self.cfg
        if c["input_topic"] != "/livox/lidar" or c["source_frame"] != "livox_frame" or c["fixed_frame"] != "map":
            raise ValueError("this adapter requires G1 raw Livox data and floor-aligned map TF")
        bounds = {"rate_hz": (0.1, 2.0), "max_points": (100, 50000),
                  "max_range": (0.2, 2.5), "min_range": (0.0, 0.5),
                  "ground_cutoff": (0.0, 0.3), "max_receive_age": (0.1, 2.0),
                  "max_output_age": (0.2, 3.0), "display_hold_seconds": (1.0, 5.0),
                  "cell_ttl_seconds": (5.0, 60.0),
                  "max_octomap_nodes": (1000, 1000000), "snapshot_every": (1, 60)}
        for key, (low, high) in bounds.items():
            if not np.isfinite(c[key]) or not low <= c[key] <= high:
                raise ValueError("invalid " + key)
        if not c["max_receive_age"] < c["max_output_age"] < c["display_hold_seconds"]:
            raise ValueError("require receive age < output age < stale-display hold timeout")
        if not np.isfinite(c["floor_z"]) or abs(c["floor_z"]) > 10:
            raise ValueError("invalid floor_z")
        if abs(c["floor_z"]/RESOLUTION - round(c["floor_z"]/RESOLUTION)) > 1e-6:
            raise ValueError("floor_z must align to the 4 cm OctoMap lattice")
        origin = np.asarray(c["sensor_origin_body"])
        if origin.shape != (3,) or not np.isfinite(origin).all() or np.linalg.norm(origin) > 0.2:
            raise ValueError("invalid LiDAR origin in IMU/body")

    def receive(self, message):
        received = time.monotonic()
        try:
            packet = decode_livox_cdr(message)
            if packet.frame != self.cfg["source_frame"]:
                raise ValueError("unexpected LiDAR frame")
        except (ValueError, UnicodeError) as exc:
            self.input_error = str(exc)
            return
        self.scans.append(Scan(packet, received, packet.start_ns, packet.end_ns))
        self.input_error = None

    def schedule(self):
        if self.pending is None or self.pending.done():
            self.pending = self.worker.submit(self.tick)

    def publish_cloud(self, name, points, stamp):
        msg = PointCloud2()
        msg.header.frame_id, msg.header.stamp = "map", stamp
        msg.height, msg.width = 1, len(points)
        msg.fields = [PointField(name=n, offset=i*4, datatype=7, count=1)
                      for i, n in enumerate(("x", "y", "z"))]
        msg.is_bigendian, msg.is_dense = False, True
        msg.point_step, msg.row_step = 12, len(points)*12
        msg.data = np.ascontiguousarray(points, dtype="<f4").tobytes()
        self.cloud_pubs[name].publish(msg)

    def report(self, state, **details):
        data = dict(state=state, perception_only=True, motion_enabled=False,
                    navigation_arm_state="not_monitored",
                    policy_ready=False, data_valid=(state == "PREVIEW_OK"),
                    sequence=self.sequence, input_topic=self.cfg["input_topic"],
                    frame="map", shape=list(SHAPE), resolution=RESOLUTION, **details)
        message = String()
        message.data = json.dumps(data, sort_keys=True, allow_nan=False)
        self.status_pub.publish(message)
        if state != self.last_status or time.monotonic() - self.last_report_time >= 5:
            print(message.data, flush=True)
            self.last_report_time = time.monotonic()
        self.last_status = state

    def invalidate(self, reason, hard=False):
        # A short TF/input gap does not erase observations or pretend to be fresh.
        # Existing cloud timestamps remain unchanged; amber ROI labels a held view.
        age = (time.monotonic() - self.last_good_received
               if self.last_good_received is not None else None)
        if not hard and age is not None and age < self.cfg["display_hold_seconds"]:
            self.publish_roi(self.last_origin, self.get_clock().now().to_msg(), stale=True)
            self.report("HOLDING_STALE", reason=reason, display_age_s=age)
            return
        self.tree.reset()
        self.last_pose = self.last_origin = None
        self.last_good_received = None
        self.reset_time = time.monotonic()
        stamp = self.get_clock().now().to_msg()
        for name in CLOUD_TOPICS:
            self.publish_cloud(name, np.empty((0, 3)), stamp)
        marker = Marker()
        marker.header.frame_id, marker.ns, marker.id = "map", "cat_roi", 0
        marker.action = Marker.DELETE
        self.roi_pub.publish(marker)
        self.report("WAITING_OR_INVALID", reason=reason)

    def pose_at(self, ns):
        result = self.tf_buffer.lookup_transform("map", "body", Time(nanoseconds=int(ns)))
        t, q = result.transform.translation, result.transform.rotation
        return [t.x, t.y, t.z], [q.x, q.y, q.z, q.w]

    def select_scan(self, now):
        scans, generation = self.scans.snapshot()
        if generation != self.generation:
            self.invalidate("sensor timestamp moved backwards", hard=True)
            self.last_stamp_ns = -1
            self.generation = generation
        if not scans:
            raise ValueError(self.input_error or "waiting for raw LiDAR")
        if now - scans[-1].received > self.cfg["max_receive_age"]:
            raise ValueError("no fresh raw LiDAR receive")
        latest_tf = self.tf_buffer.lookup_transform("map", "body", Time())
        selected = ScanBuffer.select(scans, now, self.cfg["max_receive_age"],
                                     self.last_stamp_ns, stamp_ns(latest_tf.header.stamp))
        if selected is None:
            raise ValueError("waiting for sensor-time TF covering a fresh complete scan")
        # Validate both ends; all subsequent bins are within this exact interval.
        self.pose_at(selected.start_ns)
        self.pose_at(selected.end_ns)
        return selected

    def tick(self):
        started = time.monotonic()
        timing = {}
        mark = started
        def stage(name):
            nonlocal mark
            current = time.monotonic()
            timing[name] = round((current-mark)*1000, 2)
            mark = current
        try:
            scan = self.select_scan(started)
            stage("select_ms")
            xyz, offsets, info = livox_points(scan.message, self.cfg["max_points"], self.cfg["min_range"])
            stage("decode_ms")
            if len(xyz) < 10:
                raise ValueError("too few valid raw returns")
            points, origins = deskew_livox(xyz, offsets, scan.start_ns, self.pose_at,
                                           self.cfg["sensor_origin_body"])
            translation, quat = self.pose_at(scan.end_ns)
            sensor = transform_points([self.cfg["sensor_origin_body"]], translation, quat)[0]
            origin = grid_origin(translation, self.cfg["floor_z"])
            stage("deskew_ms")
            reset_reason = "none"
            if self.last_pose is not None:
                old_t, old_r = self.last_pose
                angle = np.arccos(np.clip((np.trace(old_r.T @ rotation_matrix(quat))-1)/2, -1, 1))
                if np.linalg.norm(np.asarray(translation)-old_t) > 0.5 or angle > np.deg2rad(30):
                    self.tree.reset()
                    self.reset_time = started
                    reset_reason = "large_pose_change"
            ground = points[:, 2] - self.cfg["floor_z"] < self.cfg["ground_cutoff"]
            # Bound storage before tracing: rays outside the published volume
            # must not repeatedly allocate cells only to retire them afterward.
            removed = self.tree.prune(origin, started, self.cfg["cell_ttl_seconds"])
            self.tree.insert(points, ~ground, origins, self.cfg["max_range"], observed_at=started)
            stage("integrate_ms")
            if self.tree.nodes > self.cfg["max_octomap_nodes"]:
                self.invalidate("OctoMap node budget exceeded", hard=True)
                return
            grid = self.tree.export(origin)
            raw = grid == 2
            stage("export_ms")
            processed = cat_preprocess(raw)
            stage("morphology_ms")
            ranges = np.linalg.norm(xyz, axis=1)
            within = (ranges <= self.cfg["max_range"])
            within &= (points >= origin).all(axis=1) & (points < origin + np.array(SHAPE)*RESOLUTION).all(axis=1)
            clouds = (("input_cloud", points[within]), ("ground_points", points[within & ground]),
                      ("obstacle_points", points[within & ~ground]),
                      ("nearfield_points", points[within & (ranges < 0.5)]),
                      ("occupancy_raw", centers(raw, origin)),
                      ("occupancy_cat", centers(processed, origin)))
            stage("prepare_clouds_ms")
            if time.monotonic() - scan.received > self.cfg["max_output_age"]:
                raise ValueError("processing exceeded maximum output age")
            if self.scans.snapshot()[1] != self.generation:
                self.invalidate("clock reset during processing", hard=True)
                return
            self.last_pose = (np.asarray(translation), rotation_matrix(quat))
            self.last_origin = origin.copy()
            self.last_good_received = scan.received
            self.last_stamp_ns = scan.start_ns
            self.sequence += 1
            now_msg = self.get_clock().now().to_msg()
            for name, values in clouds:
                self.publish_cloud(name, values, now_msg)
            self.publish_roi(origin, now_msg)
            details = dict(info, origin=origin.tolist(), sensor_origin=sensor.tolist(),
                           obstacle_returns=int((within & ~ground).sum()),
                           ground_returns=int((within & ground).sum()),
                           nearfield_in_volume=int((within & (ranges < 0.5)).sum()),
                           occupied_raw=int(raw.sum()), occupied_cat=int(processed.sum()),
                           cat_added=int((processed & ~raw).sum()), cat_removed=int((raw & ~processed).sum()),
                           known_free=int((grid == 1).sum()), unknown=int((grid == 0).sum()),
                           source_stamp=scan.start_ns*1e-9, source_end_stamp=scan.end_ns*1e-9,
                           receive_age_s=started-scan.received, tf_mode="point_time_10ms",
                           min_range=self.cfg["min_range"], octomap_nodes=self.tree.nodes,
                           cells_expired_or_outside=removed, cell_ttl_s=self.cfg["cell_ttl_seconds"],
                           history_age_s=started-self.reset_time, history_reset=reset_reason)
            if self.cfg["snapshot_directory"] and self.sequence % self.cfg["snapshot_every"] == 0:
                directory = Path(self.cfg["snapshot_directory"])
                directory.mkdir(parents=True, exist_ok=True)
                target, temporary = directory / "latest.npz", directory / "latest.tmp.npz"
                np.savez_compressed(str(temporary), occupancy_state=grid,
                                    cat_occupancy=processed.astype(np.uint8), origin=origin,
                                    resolution=RESOLUTION, source_stamp=scan.start_ns*1e-9,
                                    metadata=json.dumps(details))
                os.replace(str(temporary), str(target))
            details["processing_ms"] = (time.monotonic()-started)*1000
            details["output_receive_age_s"] = time.monotonic()-scan.received
            stage("publish_snapshot_ms")
            details["timings"] = timing
            if details["output_receive_age_s"] > self.cfg["max_output_age"]:
                self.invalidate("published sample aged while writing snapshot")
            else:
                self.report("PREVIEW_OK", **details)
        except Exception as exc:
            self.invalidate(type(exc).__name__ + ": " + str(exc))

    def publish_roi(self, origin, stamp, stale=False):
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = "map", stamp
        marker.ns, marker.id, marker.type, marker.action = "cat_roi", 0, Marker.LINE_LIST, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.015
        marker.color.r = 1.0 if stale else 0.0
        marker.color.g, marker.color.b, marker.color.a = 0.8, 0.0 if stale else 1.0, 0.8
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
        node.timer.cancel()
        node.worker.shutdown(wait=True)
        if rclpy.ok():
            node.invalidate("preview stopped", hard=True)
        node.tree.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
