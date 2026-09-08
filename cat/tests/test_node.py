"""Node failure gates and stale-display behavior, without any DDS participant."""
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as Obj
from unittest.mock import Mock
import unittest

try:
    from preview_node import CatPreview
except ImportError:
    CatPreview = None


@unittest.skipIf(CatPreview is None, "ROS/Livox imports unavailable in native-only test environment")
class NodeGatesTests(unittest.TestCase):
    def fake(self):
        node = Obj(cfg={"arm_token": "/nonexistent/cat-test-arm-token",
                       "display_hold_seconds": 3.0}, last_good_received=None,
                   last_pose=None, last_origin=None,
                   tree=Obj(reset=Mock()), report=Mock(), publish_roi=Mock(),
                   publish_cloud=Mock(), roi_pub=Obj(publish=Mock()))
        from builtin_interfaces.msg import Time
        node.get_clock = lambda: Obj(now=lambda: Obj(to_msg=lambda: Time(sec=100)))
        node.invalidate = lambda reason, hard=False: CatPreview.invalidate(node, reason, hard)
        return node

    def test_transient_gap_holds_display_and_history_with_stale_status(self):
        node = self.fake()
        node.last_good_received = time.monotonic() - 1.2
        node.last_origin = [0, 0, 0]
        node.invalidate("TF pending")
        node.tree.reset.assert_not_called()
        node.publish_cloud.assert_not_called()
        self.assertEqual(node.report.call_args[0][0], "HOLDING_STALE")
        self.assertTrue(node.publish_roi.call_args[1]["stale"])

    def test_real_staleness_clears_display_and_history(self):
        node = self.fake()
        node.last_good_received = time.monotonic() - 4
        node.invalidate("input stopped")
        node.tree.reset.assert_called_once()
        self.assertGreater(node.publish_cloud.call_count, 0)
        self.assertEqual(node.report.call_args[0][0], "WAITING_OR_INVALID")

    def test_arm_token_clears_immediately_without_removing_token(self):
        with tempfile.TemporaryDirectory(prefix="cat-gate-test-") as directory:
            token = Path(directory) / "armed"
            token.touch()
            node = self.fake()
            node.cfg["arm_token"] = str(token)
            node.last_good_received = time.monotonic()
            CatPreview.tick(node)
            node.tree.reset.assert_called_once()
            self.assertTrue(token.exists())

    def test_missing_tf_is_stale_not_an_identity_pose(self):
        node = self.fake()
        node.select_scan = Mock(side_effect=ValueError("no TF"))
        CatPreview.tick(node)
        self.assertIn("no TF", node.report.call_args[1]["reason"])

    def test_pose_at_asks_for_exact_sensor_timestamp(self):
        node = self.fake()
        node.tf_buffer = Obj(lookup_transform=Mock(side_effect=ValueError("missing exact TF")))
        with self.assertRaisesRegex(ValueError, "missing exact TF"):
            CatPreview.pose_at(node, 123456789000)
        args = node.tf_buffer.lookup_transform.call_args[0]
        self.assertEqual(args[:2], ("map", "body"))
        self.assertEqual(args[2].nanoseconds, 123456789000)


if __name__ == "__main__":
    unittest.main()
