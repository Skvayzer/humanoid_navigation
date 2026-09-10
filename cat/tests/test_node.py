"""Node failure gates and stale-display behavior, without any DDS participant."""
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as Obj
from unittest.mock import Mock, patch
import unittest
import numpy as np

try:
    from preview_node import CatPreview
except ImportError:
    CatPreview = None


@unittest.skipIf(CatPreview is None, "ROS/Livox imports unavailable in native-only test environment")
class NodeGatesTests(unittest.TestCase):
    def fake(self):
        node = Obj(cfg={"display_hold_seconds": 3.0}, last_good_received=None,
                   last_pose=None, last_origin=None,
                   tree=Obj(reset=Mock()), report=Mock(), publish_roi=Mock(),
                   publish_cloud=Mock(), roi_pub=Obj(publish=Mock()),
                   on_sample=Mock(), on_invalid=Mock())
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

    def test_armed_navigation_does_not_gate_processing_or_change_token(self):
        with tempfile.TemporaryDirectory(prefix="cat-gate-test-") as directory:
            token = Path(directory) / "g1_motion_armed"
            token.touch()
            node = self.fake()
            # Even a legacy arm_token setting must not gate CAT at either of
            # the former checks (before processing and before publication).
            node.cfg.update(arm_token=str(token), max_points=20000, min_range=0.,
                            sensor_origin_body=[0.,0.,0.], floor_z=0., ground_cutoff=.1,
                            cell_ttl_seconds=30., max_range=2.5, max_octomap_nodes=1000,
                            max_output_age=2., snapshot_directory='')
            node.reset_time = time.monotonic()
            node.sequence, node.generation = 0, 0
            node.scans = Obj(snapshot=lambda: ([], 0))
            node.select_scan = Mock(return_value=Obj(message=None, received=time.monotonic(),
                                                     start_ns=1000000000, end_ns=1100000000,
                                                     packet_count=1))
            node.pose_at = Mock(return_value=([0.,0.,0.], [0.,0.,0.,1.]))
            node.tree.nodes = 10
            node.tree.prune, node.tree.insert = Mock(return_value=0), Mock()
            node.tree.export = Mock(return_value=np.zeros((128,128,35), dtype=np.uint8))
            xyz = np.tile([.2,0.,.2], (12,1))
            with patch('preview_node.Path.exists', side_effect=AssertionError('must not inspect arming')), \
                 patch('preview_node.livox_points', return_value=(xyz, np.zeros(12), {})), \
                 patch('preview_node.deskew_livox', return_value=(xyz, np.zeros_like(xyz))):
                CatPreview.tick(node)
            node.select_scan.assert_called_once()
            self.assertEqual(node.report.call_args[0][0], 'PREVIEW_OK')
            self.assertEqual(node.sequence, 1)
            self.assertEqual(node.publish_cloud.call_count, 6)
            node.tree.reset.assert_not_called()
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
