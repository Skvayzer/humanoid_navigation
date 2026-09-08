#!/usr/bin/env python3
"""Fail-closed Nav2 Twist to Unitree G1 high-level locomotion gateway.

ROS and Unitree SDK participants run in separate processes so their Cyclone
DDS runtimes cannot interfere. This never changes the robot FSM or motion mode.
"""

import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from pathlib import Path

CMD_TOPIC = "/g1_nav/cmd_vel_preview"
LIMITED_TOPIC = "/g1_nav/cmd_vel_limited"
ARMED_TOPIC = "/g1_nav/motion_bridge/armed"
RUNTIME_DIR = Path(os.environ.get("G1_SLAM_RUNTIME_DIR", str(Path.home() / "g1_slam_runtime")))
ARM_FILE = RUNTIME_DIR / "nav/g1_motion_armed"
LOG_FILE = RUNTIME_DIR / "logs/nav/g1_motion_bridge.log"
SDK_PATH = os.environ.get("G1_SDK_PATH", str(Path(__file__).resolve().parents[1] / "third_party/unitree_sdk2_python"))
SDK_INTERFACE = os.environ.get("G1_SDK_INTERFACE", "eth0")

CONTROL_HZ = 10.0
CMD_TIMEOUT_S = 0.30
SDK_COMMAND_DURATION_S = 0.20
SDK_QUEUE_TIMEOUT_S = 0.16

MAX_VX = 0.20
MAX_WZ = 0.20
MAX_DVX_PER_S = 0.10
MAX_DWZ_PER_S = 0.20
FIRST_MOTION_S = 3.0
# Keep the first-motion hook from the X2 design, but do not suppress commands
# below the G1 gait's observed useful range. Slew limiting still ramps them.
FIRST_MAX_VX = 0.20
FIRST_MAX_WZ = 0.20
ARM_SETTLE_S = 1.0

def _append_log(message):
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as stream:
            stream.write("{} {}\n".format(time.strftime("%Y-%m-%dT%H:%M:%S"), message))
    except OSError:
        pass


def _finite_clamp(value, low, high):
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(low, min(high, value))


def _put_latest(command_queue, item):
    while True:
        try:
            command_queue.get_nowait()
        except queue.Empty:
            break
    try:
        command_queue.put_nowait(item)
    except queue.Full:
        pass


def sdk_worker(command_queue, stop_event):
    """Own the Unitree SDK participant and expire commands independently."""
    os.environ.pop("CYCLONEDDS_URI", None)
    if SDK_PATH not in sys.path:
        sys.path.insert(0, SDK_PATH)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    ChannelFactoryInitialize(0, SDK_INTERFACE)
    client = LocoClient()
    client.SetTimeout(0.25)
    client.Init()
    _append_log("SDK ready on {}; no FSM/mode command sent".format(SDK_INTERFACE))

    last_rx = time.monotonic()
    vx = 0.0
    wz = 0.0
    last_report = None
    try:
        while not stop_event.is_set():
            try:
                item = command_queue.get(timeout=SDK_QUEUE_TIMEOUT_S)
                while True:
                    try:
                        item = command_queue.get_nowait()
                    except queue.Empty:
                        break
                vx, wz = item
                last_rx = time.monotonic()
            except queue.Empty:
                vx, wz = 0.0, 0.0
            if time.monotonic() - last_rx > CMD_TIMEOUT_S:
                vx, wz = 0.0, 0.0

            code = client.SetVelocity(vx, 0.0, wz, SDK_COMMAND_DURATION_S)
            report = (round(vx, 3), round(wz, 3), code)
            if report != last_report:
                _append_log("SDK velocity vx={:.3f} vy=0.000 wz={:.3f} code={}".format(vx, wz, code))
                last_report = report
    except BaseException as exc:
        _append_log("SDK worker exception: {!r}".format(exc))
        raise
    finally:
        for _ in range(5):
            try:
                client.SetVelocity(0.0, 0.0, 0.0, SDK_COMMAND_DURATION_S)
            except BaseException:
                pass


def main():
    # Spawn provides a fresh interpreter and avoids inherited rclpy/DDS state.
    ctx = mp.get_context("spawn")
    command_queue = ctx.Queue(maxsize=2)
    stop_event = ctx.Event()
    sdk_process = ctx.Process(
        target=sdk_worker,
        args=(command_queue, stop_event),
        name="g1_unitree_sdk_gateway",
    )
    sdk_process.start()

    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from std_msgs.msg import Bool

    class Bridge(Node):
        def __init__(self):
            super().__init__("g1_cmd_vel_safety_bridge")
            self.latest = None
            self.latest_time = 0.0
            self.was_armed = False
            self.armed_since = 0.0
            self.motion_since = None
            self.last_vx = 0.0
            self.last_wz = 0.0
            self.last_tick = time.monotonic()
            self.last_state = None
            self.audit_origin = None
            self.audit_last_logged_distance = 0.0
            self.limited_pub = self.create_publisher(Twist, LIMITED_TOPIC, 10)
            self.armed_pub = self.create_publisher(Bool, ARMED_TOPIC, 10)
            self.create_subscription(Twist, CMD_TOPIC, self.on_cmd, 10)
            self.create_subscription(Odometry, "/g1_nav/odom", self.on_odom, 10)
            self.create_timer(1.0 / CONTROL_HZ, self.tick)
            self.get_logger().warning(
                "DISARMED: no nonzero Unitree velocity can pass until g1-motion-arm"
            )
            _append_log(
                "bridge started DISARMED; limits vx<=0.20, |wz|<=0.20, "
                "vy=0, reverse=off, watchdog=0.30s, steering=combined"
            )

        def on_cmd(self, msg):
            self.latest = msg
            self.latest_time = time.monotonic()

        @staticmethod
        def _wrap_angle(angle):
            return math.atan2(math.sin(angle), math.cos(angle))

        def on_odom(self, msg):
            """Passive forward-axis audit; this callback never commands motion."""
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            if self.audit_origin is None:
                self.audit_origin = (float(p.x), float(p.y), yaw)
                self.audit_last_logged_distance = 0.0
                _append_log(
                    "HEADING_AUDIT origin x={:.4f} y={:.4f} nav_yaw_deg={:.2f}".format(
                        p.x, p.y, math.degrees(yaw)
                    )
                )
                return

            x0, y0, yaw0 = self.audit_origin
            dx = float(p.x) - x0
            dy = float(p.y) - y0
            distance = math.hypot(dx, dy)
            if distance < 0.05 or distance - self.audit_last_logged_distance < 0.05:
                return
            travel_yaw = math.atan2(dy, dx)
            error = self._wrap_angle(travel_yaw - yaw0)
            _append_log(
                "HEADING_AUDIT distance={:.3f} dx={:.3f} dy={:.3f} "
                "travel_yaw_deg={:.2f} origin_nav_yaw_deg={:.2f} error_deg={:.2f}".format(
                    distance,
                    dx,
                    dy,
                    math.degrees(travel_yaw),
                    math.degrees(yaw0),
                    math.degrees(error),
                )
            )
            self.audit_last_logged_distance = distance

        def tick(self):
            now = time.monotonic()
            dt = max(0.0, min(0.25, now - self.last_tick))
            self.last_tick = now
            armed = ARM_FILE.is_file()

            if armed and not self.was_armed:
                self.armed_since = now
                self.motion_since = None
                self.latest = None
                self.get_logger().warning(
                    "ARM token detected; holding zero for {:.1f}s and waiting for a fresh command".format(ARM_SETTLE_S)
                )
                _append_log("ARMED transition; settling at zero")
            elif not armed and self.was_armed:
                self.get_logger().warning("DISARMED; commanding zero")
                _append_log("DISARMED transition")
            self.was_armed = armed

            fresh = self.latest is not None and now - self.latest_time <= CMD_TIMEOUT_S
            enabled = armed and now - self.armed_since >= ARM_SETTLE_S and fresh
            target_vx = 0.0
            target_wz = 0.0
            if enabled:
                target_vx = _finite_clamp(self.latest.linear.x, 0.0, MAX_VX)
                target_wz = _finite_clamp(self.latest.angular.z, -MAX_WZ, MAX_WZ)

                moving = abs(target_vx) > 1.0e-4 or abs(target_wz) > 1.0e-4
                if moving and self.motion_since is None:
                    self.motion_since = now
                if not moving:
                    self.motion_since = None
                if self.motion_since is not None and now - self.motion_since < FIRST_MOTION_S:
                    target_vx = min(target_vx, FIRST_MAX_VX)
                    target_wz = _finite_clamp(target_wz, -FIRST_MAX_WZ, FIRST_MAX_WZ)
            else:
                self.motion_since = None

            # Disarm/stale-command stops are immediate. While enabled, apply
            # independent slew limits to DWB's combined forward and yaw command.
            if enabled:
                dv = MAX_DVX_PER_S * dt
                dw = MAX_DWZ_PER_S * dt
                vx = _finite_clamp(target_vx, self.last_vx - dv, self.last_vx + dv)
                wz = _finite_clamp(target_wz, self.last_wz - dw, self.last_wz + dw)
            else:
                vx, wz = 0.0, 0.0
            self.last_vx, self.last_wz = vx, wz

            _put_latest(command_queue, (vx, wz))
            limited = Twist()
            limited.linear.x = vx
            limited.angular.z = wz
            self.limited_pub.publish(limited)
            state = Bool()
            state.data = armed
            self.armed_pub.publish(state)

            summary = (armed, fresh, round(vx, 3), round(wz, 3))
            if summary != self.last_state:
                _append_log(
                    "state armed={} fresh={} vx={:.3f} wz={:.3f}".format(
                        armed, fresh, vx, wz
                    )
                )
                self.last_state = summary

    rclpy.init()
    node = Bridge()

    def request_shutdown(_signum, _frame):
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        rclpy.spin(node)
    finally:
        _put_latest(command_queue, (0.0, 0.0))
        stop_event.set()
        sdk_process.join(timeout=2.0)
        if sdk_process.is_alive():
            sdk_process.terminate()
            sdk_process.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _append_log("bridge stopped")


if __name__ == "__main__":
    main()
