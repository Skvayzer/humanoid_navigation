"""Exercise node failure gates without starting a ROS/DDS participant."""
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as Obj
import unittest

import numpy as np

try:
    from preview_node import CatPreview
except ImportError:
    CatPreview = None


@unittest.skipIf(CatPreview is None, "ROS imports unavailable in native-only test environment")
class NodeGatesTests(unittest.TestCase):
    def fake(self):
        # No Node constructor, publishers or transport. Failures occur before TF.
        cloud = Obj(header=Obj(stamp=Obj(sec=100, nanosec=0), frame_id="body"))
        node = Obj(cfg={"arm_token": "/nonexistent/cat-test-arm-token",
                       "input_topic": "/g1_slam/cloud_registered_body",
                       "source_frame": "body", "max_receive_age": 0.5},
                   latest=(cloud, time.monotonic()), last_stamp=None, reason=None)
        node.invalidate = lambda reason: setattr(node, "reason", reason)
        return node

    def test_missing_scan(self):
        node = self.fake()
        node.latest = None
        CatPreview.tick(node)
        self.assertIn("waiting for", node.reason)

    def test_stale_scan(self):
        node = self.fake()
        node.latest = (node.latest[0], time.monotonic()-2)
        CatPreview.tick(node)
        self.assertIn("stale", node.reason)

    def test_repeated_stamp_stays_rejected(self):
        node = self.fake()
        node.last_stamp = 100
        for _ in range(2):
            CatPreview.tick(node)
            self.assertIn("timestamp stopped", node.reason)
            self.assertEqual(node.last_stamp, 100)

    def test_backwards_stamp_requires_advancing_new_scan(self):
        node = self.fake()
        node.last_stamp = 200
        CatPreview.tick(node)
        self.assertEqual(node.last_stamp, 100)
        CatPreview.tick(node)
        self.assertIn("timestamp stopped", node.reason)

    def test_wrong_frame(self):
        node = self.fake()
        node.latest[0].header.frame_id = "livox_frame"
        CatPreview.tick(node)
        self.assertIn("unexpected source frame", node.reason)

    def test_arm_token_blocks_even_if_cloud_is_available(self):
        with tempfile.TemporaryDirectory(prefix="cat-gate-test-") as directory:
            token = Path(directory) / "armed"
            token.touch()
            node = self.fake()
            node.cfg["arm_token"] = str(token)
            CatPreview.tick(node)
            self.assertIn("arm token exists", node.reason)
            self.assertTrue(token.exists())  # preview never disarms another controller

    def test_missing_tf_has_no_identity_fallback(self):
        node = self.fake()
        def unavailable(_):
            raise ValueError("no TF")
        node.lookup = unavailable
        CatPreview.tick(node)
        self.assertIn("no TF", node.reason)

    def test_latest_tf_too_old_is_rejected(self):
        def lookup(*_):
            return Obj(header=Obj(stamp=Obj(sec=99, nanosec=0)))
        node = self.fake()
        node.cfg["max_tf_delta"] = 0.25
        node.tf_buffer = Obj(lookup_transform=lookup)
        # Use generated Time rather than fake message (Time.from_msg validates).
        from builtin_interfaces.msg import Time
        node.latest[0].header.stamp = Time(sec=100)
        with self.assertRaisesRegex(ValueError, "TF is stale"):
            CatPreview.lookup(node, node.latest[0])


if __name__ == "__main__":
    unittest.main()
