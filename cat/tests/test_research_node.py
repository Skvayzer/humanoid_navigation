"""ROS-facing research gates; no connection to the robot for unit tests."""
import os
from types import SimpleNamespace as Obj
from unittest.mock import Mock, patch
import threading
import time
import unittest
import numpy as np

try:
    from research_node import ResearchPreview, validate_goal
    from geometry_msgs.msg import PoseStamped
except ImportError:
    ResearchPreview = None


@unittest.skipIf(ResearchPreview is None, 'ROS imports unavailable')
class ResearchNodeTests(unittest.TestCase):
    def test_goal_is_dedicated_map_xy_not_arbitrary_frame(self):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.orientation.w = 1.
        goal.pose.position.x, goal.pose.position.y = 1., 2.
        goal.pose.position.z = 0.
        np.testing.assert_allclose(validate_goal(goal, 0.), [1, 2, .75])
        goal.header.frame_id = 'body'
        with self.assertRaises(ValueError):
            validate_goal(goal, 0.)

    def test_bad_goal_cancels_old_goal_and_fields(self):
        node = Obj(field_lock=threading.RLock(), cfg=dict(floor_z=0.),
                   goal=[1, 1, 1], goal_received=1., clear_fields=Mock())
        ResearchPreview.receive_goal(node, PoseStamped())
        self.assertIsNone(node.goal)
        node.clear_fields.assert_called_once()

    def test_stale_input_never_publishes_new_fields(self):
        node = Obj(field_lock=threading.RLock(), field_epoch=2, cfg=dict(max_output_age=2.),
                   clear_fields=Mock(), publish_slice=Mock())
        scan = Obj(received=time.monotonic()-3)
        ResearchPreview.compute_and_publish(node, None, None, None, scan, None, 2, 1)
        node.publish_slice.assert_not_called()
        node.clear_fields.assert_called_once()

    def test_superseded_failure_does_not_erase_new_fields(self):
        node = Obj(field_lock=threading.RLock(), field_epoch=3, cfg=dict(max_output_age=2.),
                   clear_fields=Mock(), publish_slice=Mock())
        scan = Obj(received=time.monotonic()-3)
        ResearchPreview.compute_and_publish(node, None, None, None, scan, None, 2, 1)
        node.clear_fields.assert_not_called()

    def test_timeout_clears_fields_and_goal_expires(self):
        node = Obj(field_lock=threading.RLock(), field_received=time.monotonic()-3,
                   goal_received=None, cfg=dict(max_output_age=2.), clear_fields=Mock())
        ResearchPreview.check_field_age(node)
        node.clear_fields.assert_called_once()
        node.clear_fields.reset_mock()
        node.goal_received = time.monotonic()-61
        node.goal = np.ones(3)
        ResearchPreview.check_field_age(node)
        self.assertIsNone(node.goal)
        node.clear_fields.assert_called_once()


@unittest.skipUnless(os.environ.get('CAT_RUN_ROS_GRAPH_TESTS') == '1', 'isolated ROS graph test opt-in only')
class ResearchWiringTests(unittest.TestCase):
    def test_real_lowstate_subscription_and_visualization_only_publishers(self):
        self.assertEqual(os.environ.get('ROS_DOMAIN_ID'), '101')
        import rclpy
        from unitree_hg.msg import LowState
        rclpy.init(args=[])
        node = None
        try:
            node = ResearchPreview()
            for timer in (node.timer, node.state_timer, node.health_timer):
                timer.cancel()
            topics = {sub.topic_name for sub in node.subscriptions}
            self.assertTrue({'/lowstate', '/livox/lidar', '/tf', '/tf_static', '/g1_cat/goal_preview'} <= topics)
            self.assertNotIn('/goal_pose', topics)
            for publisher in node.publishers:
                self.assertTrue(publisher.topic_name.startswith('/g1_cat/') or publisher.topic_name == '/parameter_events')
            # Synthetic input only in an isolated test domain, never on domain 0.
            publisher = node.create_publisher(LowState, '/lowstate', 1)
            msg = LowState()
            msg.tick = 1
            msg.imu_state.quaternion = [1., 0., 0., 0.]
            deadline = time.monotonic()+3.
            while node.state_buffer.state is None and time.monotonic() < deadline:
                publisher.publish(msg)
                rclpy.spin_once(node, timeout_sec=.05)
            self.assertIsNotNone(node.state_buffer.state, node.state_error)
            # Exercise ROS message construction/serialization too.
            with patch.object(node, 'status') as status:
                node.publish_state()
                self.assertEqual(status.call_args[1]['state'], 'TELEMETRY_OK_UNCALIBRATED')
        finally:
            if node is not None:
                node.worker.shutdown(wait=True)
                node.field_worker.shutdown(wait=True)
                node.tree.close()
                node.destroy_node()
            rclpy.shutdown()
