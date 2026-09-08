import unittest
from types import SimpleNamespace as Obj

import numpy as np

from perception_core import Scan, ScanBuffer, livox_points, deskew_livox


def packet(points):
    return Obj(point_num=len(points), points=[Obj(x=x, y=y, z=z, offset_time=t, tag=tag, line=0)
                                             for x, y, z, t, tag in points])


class StreamTests(unittest.TestCase):
    def test_selects_newest_fresh_complete_scan_covered_by_tf(self):
        scans = [Scan(None, 1.0, 100, 200), Scan(None, 1.1, 200, 300), Scan(None, 1.2, 300, 400)]
        selected = ScanBuffer.select(scans, now=1.5, max_age=1, after_ns=100, tf_end_ns=350)
        self.assertEqual(selected.start_ns, 200)
        self.assertIsNone(ScanBuffer.select(scans, 3, 1, 0, 500))
        self.assertIsNone(ScanBuffer.select(scans, 1.5, 1, 300, 350))

    def test_queue_is_bounded_duplicate_ignored_clock_reset_explicit(self):
        buf = ScanBuffer(capacity=2)
        for i in range(4):
            buf.append(Scan(None, i, i*100, i*100+50))
        self.assertEqual(len(buf.snapshot()[0]), 2)
        buf.append(Scan(None, 4, 300, 350))
        self.assertEqual(len(buf.snapshot()[0]), 2)
        buf.append(Scan(None, 5, 0, 50))
        self.assertEqual(len(buf.snapshot()[0]), 1)
        self.assertEqual(buf.snapshot()[1], 1)

    def test_zero_blind_retains_close_returns_but_rejects_invalid(self):
        msg = packet([(0, 0, 0, 0, 0), (.03, 0, 0, 1, 0), (.2, 0, 0, 2, 0x10),
                      (float('nan'), 0, 0, 3, 0), (.1, 0, 0, 4, 0x30), (1, 0, 0, 5, 0)])
        xyz, offsets, stats = livox_points(msg)
        np.testing.assert_allclose(xyz[:, 0], [.03, .2, 1])
        self.assertEqual(stats['invalid_returns'], 3)
        self.assertEqual(stats['nearfield_retained'], 2)
        np.testing.assert_array_equal(offsets, [1, 2, 5])

    def test_close_returns_take_precedence_over_far_sampling(self):
        msg = packet([(x, 0, 0, i, 0) for i, x in enumerate([1, 2, 3, .12, .25])])
        xyz, _, info = livox_points(msg, max_points=2)
        np.testing.assert_allclose(xyz[:, 0], [.12, .25])
        self.assertEqual(info['nearfield_retained'], 2)

    def test_bad_packet_duration_rejected(self):
        with self.assertRaisesRegex(ValueError, 'point times'):
            livox_points(packet([(1, 0, 0, 300000000, 0)]))

    def test_timed_extrinsics_and_ray_origins(self):
        # Point 1 at 0 ms, point 2 at 100 ms while body moves forward at 1 m/s.
        # The raw roll flip is applied once, before body-to-map transform.
        def pose(stamp):
            return [stamp/1e9, 0, 1], [0, 0, 0, 1]
        points, origins = deskew_livox([[1, 2, 1], [1, 2, 1]], [0, 100000000], 0,
                                      pose, [.01, 0, .02])
        np.testing.assert_allclose(points, [[1.01, -2, .02], [1.11, -2, .02]], atol=1e-6)
        np.testing.assert_allclose(origins, [[.01, 0, 1.02], [.11, 0, 1.02]], atol=1e-6)

    def test_unsorted_offsets_supported_no_latest_pose_substitution(self):
        calls = []
        def pose(stamp):
            calls.append(stamp)
            return [0, 0, 0], [0, 0, 0, 1]
        deskew_livox([[1, 0, 0]]*3, [19000000, 0, 11000000], 1000000000,
                    pose, [0, 0, 0])
        self.assertEqual(calls, [1000000000, 1015000000])


if __name__ == '__main__':
    unittest.main()
