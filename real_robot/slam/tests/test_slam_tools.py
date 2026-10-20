import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from slam_common import T_EXT, fit_floor, gravity_frame, level_transform, read_pcd, write_pcd
from floor_anchor import anchor_from_samples
from map_builder import Accumulator


def rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis); k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


class SlamToolsTest(unittest.TestCase):
    def test_level_transform_flattens_plane(self):
        rng = np.random.default_rng(0)
        a, b, c = 0.02, -0.015, 0.37
        xy = rng.uniform(-5, 5, (2000, 2)); pts = np.column_stack([xy, a * xy[:, 0] + b * xy[:, 1] + c])
        t = level_transform(a, b, c); out = pts @ t[:3, :3].T + t[:3, 3]
        self.assertLess(np.abs(out[:, 2]).max(), 1e-9)

    def test_pcd_round_trip(self):
        rng = np.random.default_rng(1); xyz = rng.normal(size=(500, 3)).astype(np.float32); inten = rng.uniform(0, 9, 500)
        with tempfile.TemporaryDirectory() as d:
            write_pcd(Path(d) / 'm.pcd', xyz, inten); back = read_pcd(Path(d) / 'm.pcd')
        np.testing.assert_array_equal(np.column_stack([back['x'], back['y'], back['z']]), xyz)
        np.testing.assert_allclose(back['intensity'], inten.astype(np.float32))

    def test_accumulator_one_point_per_voxel(self):
        acc = Accumulator(voxel=0.1)
        acc.add(np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.5, 0.5, 0.5]]))
        acc.add(np.array([[0.03, 0.03, 0.03]]))
        pts, hits = acc.merged()
        self.assertEqual(len(pts), 2); self.assertEqual(sorted(hits.tolist()), [1, 2])

    def test_floor_anchor_on_synthetic_upside_down_sensor(self):
        # map truth: IMU 1.25 m above a flat floor, robot tilted 8 deg forward, sensor mounted upside down
        rng = np.random.default_rng(2)
        r_map_body = rot([0, 1, 0], math.radians(8)) @ np.diag([1., -1., -1.])
        height = 1.25
        floor = np.column_stack([rng.uniform(-5, 5, (40000, 2)), np.zeros(40000)])
        floor = floor[np.hypot(floor[:, 0], floor[:, 1]) > 0.8]
        pts_body = (floor - [0, 0, height]) @ r_map_body          # map -> body (rows of R^T)
        lidar = pts_body - T_EXT
        up_body = r_map_body.T @ np.array([0., 0., 1.])
        acc = np.tile(up_body, (600, 1)) + rng.normal(0, 0.002, (600, 3))
        t, info = anchor_from_samples(acc, rng.normal(0, 0.005, (600, 3)), [lidar[i::20] for i in range(20)])
        self.assertAlmostEqual(info['height_m'], height, places=3)
        np.testing.assert_allclose(t[2, :3], r_map_body[2], atol=5e-3)            # gravity axis recovered
        self.assertAlmostEqual(info['imu_tilt_deg'], 8.0, delta=0.2)              # tilt from vertical, mount-independent

    def test_floor_anchor_rejects_motion(self):
        rng = np.random.default_rng(3)
        with self.assertRaises(ValueError):
            anchor_from_samples(np.tile([0, 0, -1.], (600, 1)), np.full((600, 3), 0.5), [rng.normal(size=(100, 3))])


if __name__ == '__main__':
    unittest.main()
