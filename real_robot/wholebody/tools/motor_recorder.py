#!/usr/bin/env python3
"""Full-rate motor log: every /lowstate sample (~1 kHz: q, dq, tau_est, temperature, motor mode, pelvis IMU) and every
/lowcmd message (targets, dq, kp, kd, feed-forward tau, mode). Read-only: it never publishes.

  python3 motor_recorder.py <out_dir> [seconds]     (in the executor image; g1.sh record-motors starts it)

Writes <out_dir>/motors_<unix ms>.npz every 10 s (whole files only), so a dead battery loses at most 10 s.
"""
import signal
import sys
import time
from pathlib import Path

import numpy as np

from g1_wholebody.lowlevel import NUM_BODY, decode_lowstate

BLOCK_S = 10.0


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from rclpy.impl.implementation_singleton import rclpy_implementation
    from unitree_hg.msg import LowState as LowStateMsg, LowCmd as LowCmdMsg
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else float('inf')
    rclpy.init(); node = Node('g1_motor_recorder', namespace='g1_wb'); log = node.get_logger()
    deep = QoSProfile(depth=2000, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
    low_sub = node.create_subscription(LowStateMsg, '/lowstate', lambda _: None, deep, raw=True)
    take = rclpy_implementation.rclpy_take
    state, cmd = {}, {}

    def add(store, **values):
        for k, v in values.items():
            store.setdefault(k, []).append(v)

    def on_cmd(m):
        mc = m.motor_cmd[:NUM_BODY]
        add(cmd, t=time.time(), q=[c.q for c in mc], dq=[c.dq for c in mc], kp=[c.kp for c in mc], kd=[c.kd for c in mc],
            tau=[c.tau for c in mc], mode=[c.mode for c in mc])
    node.create_subscription(LowCmdMsg, '/lowcmd', on_cmd, QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))

    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1)); signal.signal(signal.SIGTERM, lambda *_: stop.append(1))

    def flush():
        if not state and not cmd:
            return
        arrays = {'state_' + k: np.asarray(v) for k, v in state.items()}
        arrays.update({'cmd_' + k: np.asarray(v) for k, v in cmd.items()})
        name = out / ('motors_%d.npz' % int(time.time() * 1000)); tmp = name.with_suffix('.tmp.npz')
        np.savez(tmp, **arrays); tmp.rename(name)
        log.info('%s: %d LowState, %d LowCmd' % (name.name, len(state.get('t', [])), len(cmd.get('t', []))))
        state.clear(); cmd.clear()

    t_end = time.time() + seconds; last = time.time()
    log.info('recording /lowstate and /lowcmd into %s' % out)
    while rclpy.ok() and not stop and time.time() < t_end:
        with low_sub.handle as capsule:
            for _ in range(2000):
                taken = take(capsule, low_sub.msg_type, True)
                if taken is None or taken[0] is None:          # Foxy raw take: (None, info) when nothing is waiting
                    break
                s = decode_lowstate(taken[0])
                add(state, t=time.time(), tick=s.tick, q=s.q, dq=s.dq, tau=s.tau, temperature=s.temperature,
                    motor_mode=s.motor_mode, quat=s.quat_wxyz, gyro=s.gyro, acc=s.acc)
        rclpy.spin_once(node, timeout_sec=0.005)
        if time.time() - last > BLOCK_S:
            last = time.time(); flush()
    flush(); node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
