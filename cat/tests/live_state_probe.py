"""Explicit, bounded read-only LowState probe; prints aggregates, no publishers."""
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from unitree_hg.msg import LowState
from state_core import StateBuffer, Kinematics


def main():
    rclpy.init()
    node = Node('cat_readonly_state_probe', namespace='g1_cat', enable_rosout=False,
                start_parameter_services=False)
    buffer, model = StateBuffer(), Kinematics()
    accepted, errors = [], []
    def receive(message):
        try:
            if buffer.receive(message, time.monotonic()):
                model.sites(buffer.state['q'])
                accepted.append(buffer.state['tick'])
        except ValueError as exc:
            errors.append(str(exc))
    node.create_subscription(LowState, '/lowstate', receive,
                             QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
    started = time.monotonic()
    try:
        while time.monotonic()-started < 5:
            rclpy.spin_once(node, timeout_sec=.1)
        print(json.dumps(dict(accepted=len(accepted), errors=sorted(set(errors)),
                              elapsed_s=time.monotonic()-started,
                              first_tick=accepted[0] if accepted else None,
                              last_tick=accepted[-1] if accepted else None,
                              actuator_publishers=0, embodiment_verified=False)))
        if not accepted or errors:
            raise RuntimeError('telemetry/FK probe did not pass')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
