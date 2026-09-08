import ast
import json
from pathlib import Path
from types import SimpleNamespace as Obj
import unittest

import numpy as np

from core import (OccupancyMap, RESOLUTION, SHAPE, cat_preprocess, centers,
                  cloud_xyz, grid_origin, rotation_matrix, transform_points)


class GeometryTests(unittest.TestCase):
    def test_existing_body_flip_is_applied_exactly_once(self):
        points = transform_points([[1, 2, 1]], [0, 0, 1], [1, 0, 0, 0])
        np.testing.assert_allclose(points, [[1, -2, 0]])

    def test_quaternion_validation_and_normalization(self):
        np.testing.assert_allclose(rotation_matrix([0, 0, 0, 2]), np.eye(3))
        for bad in ([0, 0, 0, 0], [float("nan"), 0, 0, 1]):
            with self.assertRaises(ValueError):
                rotation_matrix(bad)

    def test_xyz_grid_floor_and_negative_coordinates(self):
        origin = grid_origin([-0.01, 1.01, 1.8])
        np.testing.assert_allclose(origin, [-2.60, -1.56, 0])
        mask = np.zeros(SHAPE, dtype=bool)
        mask[0, 1, 2] = True
        np.testing.assert_allclose(centers(mask, origin), [origin + [.02, .06, .1]], atol=1e-6)

    def test_cloud_with_offsets_endianness_and_row_padding(self):
        for endian in ("<", ">"):
            dtype = np.dtype(dict(names=["x", "y", "z"], formats=[endian+"f4"]*3,
                                  offsets=[4, 8, 12], itemsize=20))
            data = bytearray(96)
            records = np.ndarray((2, 2), dtype=dtype, buffer=data, strides=(48, 20))
            records["x"] = [[1, 2], [3, float("nan")]]
            records["y"], records["z"] = 4, 5
            msg = Obj(fields=[Obj(name=n, offset=4+i*4, datatype=7, count=1)
                              for i, n in enumerate(("x", "y", "z"))],
                      point_step=20, row_step=48, width=2, height=2,
                      is_bigendian=(endian == ">"), data=data)
            np.testing.assert_equal(cloud_xyz(msg), [[1, 4, 5], [2, 4, 5], [3, 4, 5]])
            self.assertLessEqual(len(cloud_xyz(msg, 2)), 2)
            msg.data = data[:-1]
            with self.assertRaises(ValueError):
                cloud_xyz(msg)


class CatPreprocessTests(unittest.TestCase):
    def test_empty_grid_stays_empty_no_fake_obstacles(self):
        self.assertEqual(cat_preprocess(np.zeros(SHAPE)).sum(), 0)

    def test_upward_fill_and_input_not_modified(self):
        raw = np.zeros(SHAPE, dtype=bool)
        raw[50:70, 50:70, 22:26] = True
        before = raw.copy()
        result = cat_preprocess(raw)
        self.assertTrue(result[60, 60, 34])
        self.assertFalse(raw[60, 60, 34])
        np.testing.assert_array_equal(raw, before)

    def test_boundary_thin_obstacle_loss_is_visible(self):
        raw = np.zeros(SHAPE, dtype=bool)
        raw[0, 64, 10] = True
        self.assertEqual(cat_preprocess(raw).sum(), 0)
        self.assertEqual(raw.sum(), 1)


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.tree = OccupancyMap()
        self.origin = np.array([-2.56, -2.56, 0])
        self.sensor = [.02, .02, .82]

    def tearDown(self):
        self.tree.close()

    def cell(self, grid, point):
        key = tuple(np.floor((np.array(point)-self.origin)/RESOLUTION).astype(int))
        return grid[key]

    def test_unknown_free_occupied_and_ray_clearing(self):
        self.assertTrue((self.tree.export(self.origin) == 0).all())
        self.tree.insert([[1.02, .02, .82]], [1], self.sensor)
        grid = self.tree.export(self.origin)
        self.assertEqual(self.cell(grid, [1.02, .02, .82]), 2)
        self.assertEqual(self.cell(grid, [.5, .02, .82]), 1)
        self.assertEqual(self.cell(grid, [.5, 1.02, .82]), 0)
        for _ in range(4):
            self.tree.insert([[2.02, .02, .82]], [1], self.sensor)
        self.assertEqual(self.cell(self.tree.export(self.origin), [1.02, .02, .82]), 1)

    def test_ground_return_clears_without_obstacle(self):
        self.tree.insert([[1.02, .02, .02]], [0], self.sensor)
        grid = self.tree.export(self.origin)
        self.assertEqual((grid == 2).sum(), 0)
        self.assertEqual(self.cell(grid, [1.02, .02, .02]), 1)

    def test_out_of_range_return_is_not_an_obstacle(self):
        self.tree.insert([[5.02, .02, .82]], [1], self.sensor, max_range=1.0)
        grid = self.tree.export(self.origin)
        self.assertEqual((grid == 2).sum(), 0)
        self.assertEqual(self.cell(grid, [.5, .02, .82]), 1)
        self.assertEqual(self.cell(grid, [1.5, .02, .82]), 0)

    def test_hit_wins_over_intersecting_free_ray(self):
        self.tree.insert([[1.02, .02, .82], [2.02, .02, .82]], [1, 0], self.sensor)
        self.assertEqual(self.cell(self.tree.export(self.origin), [1.02, .02, .82]), 2)

    def test_reset_and_bad_shapes(self):
        self.tree.insert([[1.02, .02, .82]], [1], self.sensor)
        self.tree.reset()
        self.assertTrue((self.tree.export(self.origin) == 0).all())
        with self.assertRaises(ValueError):
            self.tree.insert([[1, 2, 3]], [], self.sensor)
        with self.assertRaises(ValueError):
            self.tree.export([1])


class NoMotionTests(unittest.TestCase):
    def test_no_policy_sdk_control_or_tf_publisher(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("core.py", "preview_node.py"):
            source = (root / filename).read_text()
            tree = ast.parse(source)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(a.name for a in node.names)
                if isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            for forbidden in ("unitree", "gx_loco", "shared_memory", "onnxruntime", "subprocess"):
                self.assertFalse(any(forbidden in module for module in imports), forbidden)
            for forbidden in ("TransformBroadcaster", "create_client", "LowCmd", "cmd_vel"):
                self.assertNotIn(forbidden, source)
        # All explicit publishers are rooted in our visualization namespace.
        tree = ast.parse((root / "preview_node.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "create_publisher":
                topic = node.args[1]
                if isinstance(topic, ast.BinOp):
                    topic = topic.left
                self.assertIsInstance(topic, ast.Constant)
                self.assertTrue(topic.value.startswith("/g1_cat/"))


if __name__ == "__main__":
    unittest.main()
