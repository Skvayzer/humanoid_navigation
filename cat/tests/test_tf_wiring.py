"""Opt-in actual ROS wiring regression, isolated test domain only, no motion."""
import os
import time
import unittest


@unittest.skipUnless(os.environ.get("CAT_RUN_ROS_GRAPH_TESTS") == "1",
                     "actual ROS graph test requires explicit isolated-domain opt-in")
class TfWiringTests(unittest.TestCase):
    def test_cat_reads_global_tf_and_receives_transforms(self):
        # Never publish these synthetic transforms on the robot's ROS domain.
        self.assertEqual(os.environ.get("ROS_DOMAIN_ID"), "101")
        import rclpy
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from rclpy.time import Time
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage
        from preview_node import CatPreview

        rclpy.init(args=[])
        node = None
        try:
            node = CatPreview()
            node.timer.cancel()  # test no processing, snapshots or preview output
            topics = {sub.topic_name for sub in node.subscriptions}
            self.assertIn("/tf", topics)
            self.assertIn("/tf_static", topics)
            self.assertNotIn("/g1_cat/tf", topics)
            self.assertNotIn("/g1_cat/tf_static", topics)
            static_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            publisher = node.create_publisher(TFMessage, "/tf_static", static_qos)
            message = TFMessage()
            transform = TransformStamped()
            transform.header.frame_id = "map"
            transform.child_frame_id = "body"
            transform.header.stamp = node.get_clock().now().to_msg()
            transform.transform.translation.z = 1.25
            transform.transform.rotation.w = 1.0
            message.transforms = [transform]
            deadline = time.monotonic() + 3.0
            received = False
            while time.monotonic() < deadline:
                publisher.publish(message)
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.tf_buffer.can_transform("map", "body", Time()):
                    received = True
                    break
            self.assertTrue(received, "global TF did not reach CAT's actual listener")
            result = node.tf_buffer.lookup_transform("map", "body", Time())
            self.assertAlmostEqual(result.transform.translation.z, 1.25)
        finally:
            if node is not None:
                node.tree.close()
                node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
