#!/usr/bin/env python3
"""ROS 2 stand-in for the G1 on the laptop (development only): FakeG1 in real time behind the robot's topics.

  python3 -m g1_wholebody.ros_fake_robot --scene sidegap_corridor_048 --anchor 1 0.5 0.4 [--gantry]

Publishes /lowstate (unitree_hg/LowState), /g1_slam/odometry (camera_init -> body, 10 Hz, latency + noise) and
TF map -> camera_init (static, like the floor anchor), subscribes /lowcmd. Test inputs:
  /g1_fake/buttons std_msgs/String  "L2,B"  (remote buttons held; "" releases)
  /g1_fake/rope    std_msgs/Float32  extra rope length in m (>= 1 slackens it completely)
"""
import argparse
import threading
import time

import numpy as np

from wholebody_core import load_contract
from .fake_robot import FakeG1
from .fields import VirtualScene
from .frames import inverse, matrix_quat
from .kinematics import FastKinematics
from .lowlevel import LowCmd, NUM_MOTORS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', default='sidegap_corridor_048'); p.add_argument('--anchor', nargs=3, type=float, default=(1.0, 0.5, 0.4))
    p.add_argument('--gantry', action='store_true'); p.add_argument('--mount', default='unitree_urdf')
    p.add_argument('--pose-hz', type=float, default=10.); p.add_argument('--pose-latency', type=float, default=0.06)
    p.add_argument('--pose-noise', type=float, default=0.01); p.add_argument('--action-latency', type=float, default=0.010)
    p.add_argument('--lowstate-hz', type=float, default=250.)
    a = p.parse_args(argv)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String, Float32
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import StaticTransformBroadcaster
    from unitree_hg.msg import LowState as LowStateMsg, LowCmd as LowCmdMsg
    contract = load_contract(); kin = FastKinematics(contract)
    scene = VirtualScene(a.scene, anchor=tuple(a.anchor))
    gantry = None
    if a.gantry:
        hang = 2.2 - (0.79 + 0.03 + 0.42)
        gantry = dict(anchor=[scene.start[0] - 0.08, scene.start[1], 2.2], length=hang, k=3000., d=300.)
    robot = FakeG1(contract, kin, scene, start=(*scene.start, a.anchor[2]), mount=a.mount, motor_kp_scale=0.9, gantry=gantry)
    base_length = gantry['length'] if gantry else None
    rclpy.init(); node = Node('g1_fake_robot')
    lock = threading.Lock()
    low_pub = node.create_publisher(LowStateMsg, '/lowstate', QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
    odom_pub = node.create_publisher(Odometry, '/g1_slam/odometry', 10)
    t_map_ci = robot.slam_pose()                          # camera_init = the IMU pose at start (FAST-LIO convention)
    static = StaticTransformBroadcaster(node); tf = TransformStamped()
    tf.header.frame_id = 'map'; tf.child_frame_id = 'camera_init'; tf.header.stamp = node.get_clock().now().to_msg()
    q = matrix_quat(t_map_ci[:3, :3]); tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z = map(float, q)
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, t_map_ci[:3, 3])
    static.sendTransform(tf)

    def on_cmd(m):
        cmd = LowCmd(m.mode_machine); cmd.mode_pr = m.mode_pr
        for i in range(NUM_MOTORS):
            mc = m.motor_cmd[i]
            cmd.mode[i], cmd.q[i], cmd.dq[i], cmd.tau[i], cmd.kp[i], cmd.kd[i] = mc.mode, mc.q, mc.dq, mc.tau, mc.kp, mc.kd
        cmd.crc = m.crc
        with lock:
            robot.send(cmd, apply_at=robot.time + a.action_latency)

    def on_buttons(m):
        with lock:
            robot.buttons = tuple(b for b in m.data.replace(' ', '').split(',') if b)

    def on_rope(m):
        with lock:
            if robot.gantry is not None:
                robot.gantry['length'] = base_length + m.data if m.data < 1 else 10.

    node.create_subscription(LowCmdMsg, '/lowcmd', on_cmd, 10)
    node.create_subscription(String, '/g1_fake/buttons', on_buttons, 10)
    node.create_subscription(Float32, '/g1_fake/rope', on_rope, 10)
    from rclpy.executors import SingleThreadedExecutor
    spinner = SingleThreadedExecutor(); spinner.add_node(node)      # drained in the physics loop (no GIL-holding thread)
    node.get_logger().info('fake G1: scene %s, gantry %s, pose %.0f Hz +%.0f ms' % (a.scene, a.gantry, a.pose_hz, 1e3 * a.pose_latency))

    pending = []; next_pose = 0.; next_low = 0.; t0 = time.monotonic(); dt = contract['physics_dt']
    while rclpy.ok():
        with lock:
            robot.step(); now = robot.time
            if now >= next_low:
                s = robot.lowstate(); next_low = now + 1. / a.lowstate_hz
            else:
                s = None
            if now >= next_pose:
                pending.append((time.monotonic() + a.pose_latency, inverse(t_map_ci) @ robot.slam_pose(a.pose_noise, a.pose_noise), node.get_clock().now()))
                next_pose = now + 1. / a.pose_hz
        if s is not None:
            m = LowStateMsg(); m.tick = s.tick; m.mode_machine = s.mode_machine
            m.imu_state.quaternion = [float(v) for v in s.quat_wxyz]; m.imu_state.gyroscope = [float(v) for v in s.gyro]
            m.imu_state.accelerometer = [float(v) for v in s.acc]
            for i in range(29):
                ms = m.motor_state[i]; ms.mode = 1; ms.q = float(s.q[i]); ms.dq = float(s.dq[i]); ms.tau_est = float(s.tau[i])
                ms.temperature = [40, 40]
            m.wireless_remote = list(s.remote)
            low_pub.publish(m)
        while pending and pending[0][0] <= time.monotonic():
            _, t, stamp = pending.pop(0)
            o = Odometry(); o.header.frame_id = 'camera_init'; o.child_frame_id = 'body'; o.header.stamp = stamp.to_msg()
            o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, t[:3, 3])
            qq = matrix_quat(t[:3, :3]); o.pose.pose.orientation.w, o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z = map(float, qq)
            odom_pub.publish(o)
        while True:
            sleep = t0 + robot.time - time.monotonic()
            if sleep <= 0:
                break
            spinner.spin_once(timeout_sec=min(sleep, 0.001))
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
