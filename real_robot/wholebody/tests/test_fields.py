import math
import unittest

import numpy as np

from g1_wholebody import fields as fl
from wholebody_sim import Scene, DX, WINDOW
from wholebody_fields import training_fields


class FieldsTest(unittest.TestCase):
    def test_identity_anchor_matches_sim_scene(self):
        for name in ('sidegap_corridor_048', 'gap_044_exact', 'flat_walk_4m'):
            sim = Scene(name)
            mine = fl.VirtualScene(name, anchor=(sim.start[0], sim.start[1], 0.))     # start stays where it is
            first = np.array([sim.start[0] - 2.54, sim.start[1] - 2.54, DX / 2])
            np.testing.assert_array_equal(mine.occupancy(first, WINDOW), sim.occupancy(first, WINDOW))
            np.testing.assert_allclose(mine.waypoints(), sim.waypoints())

    def test_anchor_moves_geometry_rigidly(self):
        base = fl.VirtualScene('sidegap_corridor_048', anchor=(0., 0., 0.))
        moved = fl.VirtualScene('sidegap_corridor_048', anchor=(1.7, -0.4, 0.6))
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(-1, 6, 4000), rng.uniform(-2, 2, 4000), rng.uniform(0, 1.5, 4000)])
        c, s = math.cos(.6), math.sin(.6); R = np.array([[c, -s], [s, c]])
        moved_pts = pts.copy(); moved_pts[:, :2] = pts[:, :2] @ R.T + np.array([1.7, -0.4])
        np.testing.assert_array_equal(base.inside_any_box(pts), moved.inside_any_box(moved_pts))
        np.testing.assert_allclose(moved.start, [1.7, -0.4])
        self.assertTrue(base.inside_any_box(pts).any())

    def test_solve_matches_sim_update_fields(self):
        sim = Scene('sidegap_plain_047'); scene = fl.VirtualScene('sidegap_plain_047', anchor=(*sim.start, 0.))
        base = sim.start + np.array([0.37, -0.11]); goal = sim.waypoints()[0]
        # wholebody_sim.update_fields, inline
        center = np.round(base / DX) * DX
        first = np.array([center[0] - WINDOW[0] / 2 * DX + DX / 2, center[1] - WINDOW[1] / 2 * DX + DX / 2, DX / 2])
        g = np.r_[goal, .75]; lo, hi = first[:2] + .3, first[:2] + (np.array(WINDOW[:2]) - 1) * DX - .3
        g[:2] = np.clip(g[:2], lo, hi)
        expected = training_fields(sim.occupancy(first, WINDOW), first, DX, g)
        got, goal_used = fl.solve_fields(base, goal, scene.occupancy)
        np.testing.assert_array_equal(got.sdf, expected.sdf); np.testing.assert_array_equal(got.gf, expected.gf)
        np.testing.assert_array_equal(got.bf, expected.bf); np.testing.assert_allclose(goal_used, g)

    def test_worker_double_buffer(self):
        scene = fl.VirtualScene('gap_044_exact', anchor=(0.5, 0.2, 0.3))
        worker = fl.FieldWorker(scene)
        try:
            def wait():
                for _ in range(600):
                    got = worker.poll()
                    if got is not None:
                        return got
                    import time; time.sleep(0.05)
                self.fail('worker timed out: %s' % worker.last_error)
            self.assertTrue(worker.request(scene.start, scene.waypoints()[0]))
            self.assertFalse(worker.request(scene.start, scene.waypoints()[0]))     # one at a time
            first, goal = wait(); slot_a = worker.in_use
            reference, _ = fl.solve_fields(scene.start, scene.waypoints()[0], scene.occupancy)
            np.testing.assert_array_equal(first.sdf, reference.sdf)
            kept = first.sdf.copy()
            worker.request(scene.start + np.array([0.8, 0.0]), scene.waypoints()[0])
            second, _ = wait()
            self.assertNotEqual(worker.in_use, slot_a)
            np.testing.assert_array_equal(first.sdf, kept)          # the grid in use was not overwritten
            self.assertFalse(np.array_equal(second.origin, first.origin))
        finally:
            worker.close()


if __name__ == '__main__':
    unittest.main()
