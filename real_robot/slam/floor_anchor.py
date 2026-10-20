#!/usr/bin/env python3
"""Floor-aligned `map` frame for odometry-only sessions: static TF map -> camera_init, measured once at start.

FAST-LIO's camera_init is the Mid-360 IMU pose at start-up: upside down and tilted, its origin 1.2 m above the floor.
The policy needs a gravity-aligned frame with the floor at z = 0. With the robot STANDING STILL for the first
seconds this node measures: gravity in the IMU frame (/g1/imu), the floor plane in the LiDAR cloud (vendor cloud
in native axes, p_body = p_lidar + T_EXT as in fastlio_g1.yaml), and the current FAST-LIO pose. Then
    T_map_body = [R(gravity, heading of body x) | (0, 0, height above floor)],  T_map_ci = T_map_body · T_ci_body⁻¹.
Map origin: on the floor under the IMU at start; map x: the robot's forward direction at start.
"""
import json
import math
import sys
import time

import numpy as np

from slam_common import T_EXT, cloud_xyz, fit_floor, gravity_frame, inverse, matrix_to_quat, pose_to_matrix


def anchor_from_samples(acc, gyro, clouds_lidar, max_gyro=0.15, max_acc_std=0.05):
    """acc/gyro: [N, 3] IMU samples (any unit for acc), clouds_lidar: list of [M, 3] native LiDAR points.
    Returns (T_map_body 4x4, info dict). Raises ValueError if the robot moved or the floor is unclear."""
    acc, gyro = np.asarray(acc, float), np.asarray(gyro, float)
    if len(acc) < 100:
        raise ValueError('too few IMU samples')
    if np.linalg.norm(gyro, axis=1).max() > max_gyro or (acc / np.linalg.norm(acc, axis=1, keepdims=True)).std(0).max() > max_acc_std:
        raise ValueError('robot not still')
    rows = gravity_frame(acc.mean(0))                       # map <- body rotation
    body = np.concatenate(clouds_lidar) + T_EXT
    pts = body @ rows.T                                     # gravity-aligned, origin at the IMU
    a, b, c, n, rms = fit_floor(pts, z_guess=-1.2, band=0.6)
    slope = math.degrees(math.atan(math.hypot(a, b)))
    if n < 1000 or rms > 0.05 or slope > 3.0:
        raise ValueError('floor fit rejected: n=%d rms=%.3f slope=%.1f' % (n, rms, slope))
    t = np.eye(4); t[:3, :3] = rows; t[2, 3] = -c
    tilt = math.degrees(math.acos(min(1.0, abs(rows[2] @ np.array([0.0, 0.0, 1.0])))))
    return t, dict(height_m=-c, floor_points=n, floor_rms_m=rms, floor_slope_deg=slope, imu_tilt_deg=tilt)


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, PointCloud2
    from tf2_ros import StaticTransformBroadcaster
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    rclpy.init()
    node = Node('floor_anchor', namespace='g1_slam')
    log = node.get_logger()
    imu, clouds, odom = [], [], [None]
    node.create_subscription(Imu, '/g1/imu', lambda m: imu.append(([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z],
                                                                    [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])), 400)
    node.create_subscription(PointCloud2, '/utlidar/cloud_livox_mid360', lambda m: clouds.append(cloud_xyz(m)) if len(clouds) < 20 else None,
                             qos_profile_sensor_data)
    node.create_subscription(Odometry, '/g1_slam/odometry', lambda m: odom.__setitem__(0, pose_to_matrix(m.pose.pose.position, m.pose.pose.orientation)), 10)
    log.info('measuring the floor frame: keep the robot STILL ...')
    t_map_ci = None
    while rclpy.ok() and t_map_ci is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if len(imu) >= 600 and len(clouds) >= 20 and odom[0] is not None:
            try:
                acc = np.array([x[0] for x in imu[-600:]]); gyro = np.array([x[1] for x in imu[-600:]])
                t_map_body, info = anchor_from_samples(acc, gyro, clouds)
                t_map_ci = t_map_body @ inverse(odom[0])
            except ValueError as exc:
                log.warn('retrying: %s' % exc); imu.clear(); clouds.clear()
    if t_map_ci is None:
        return
    tf = TransformStamped(); tf.header.frame_id = 'map'; tf.child_frame_id = 'camera_init'
    tf.header.stamp = node.get_clock().now().to_msg()
    q = matrix_to_quat(t_map_ci[:3, :3])
    tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, t_map_ci[:3, 3])
    StaticTransformBroadcaster(node).sendTransform(tf)
    log.info('map -> camera_init fixed: IMU %.3f m above the floor, tilt %.1f deg, floor rms %.1f cm (%d pts), slope %.2f deg'
             % (info['height_m'], info['imu_tilt_deg'], 100 * info['floor_rms_m'], info['floor_points'], info['floor_slope_deg']))
    if out_path:
        with open(out_path, 'w') as f:
            json.dump(dict(info, t_map_camera_init=t_map_ci.tolist(), time=time.time()), f, indent=1)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=1.0)


if __name__ == '__main__':
    main()
