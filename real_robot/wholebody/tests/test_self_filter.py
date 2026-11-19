import unittest

import numpy as np

from g1_wholebody import esdf_layers as L
from g1_wholebody.kinematics import FastKinematics
from wholebody_core import load_contract


class SelfFilterTest(unittest.TestCase):
    """The lidar self-filter on the robot's meshes, posed at each point's own time."""

    @classmethod
    def setUpClass(cls):
        c = load_contract(); cls.kin = FastKinematics(c); cls.nominal = np.asarray(c['nominal'], float)
        cls.surface = L.BodySurface(cls.kin)
        d = np.load(L.BODY_POINTS); cls.names = [str(n) for n in d['names']]; cls.pts, cls.body = d['points'], d['body']

    def surface_points(self, q, pos=(0., 0., .78), quat=(1., 0., 0., 0.), bodies=None, every=7):
        frames, _ = self.surface.posed(q, pos, quat); out = []
        for i, n in enumerate(self.names):
            if n in frames and (bodies is None or n in bodies):
                p = self.pts[self.body == i][::every].astype(float); T = frames[n]; out.append(p @ T[:3, :3].T + T[:3, 3])
        return np.vstack(out)

    def history(self, qs, t0=100.):
        return dict(t=t0 + .1 * np.arange(len(qs)), q=np.array(qs), pos=np.tile([0., 0., .78], (len(qs), 1)),
                    quat=np.tile([1., 0., 0., 0.], (len(qs), 1)))

    def test_own_surface_removed_objects_kept(self):
        rng = np.random.default_rng(0)
        own = self.surface_points(self.nominal) + rng.normal(0, .01, (1, 3))
        h = self.history([self.nominal])
        self.assertGreater(self.surface.inside(own, h).mean(), .97)
        # an object 10 cm in front of the right palm, and a wall 0.5 m ahead
        frames, palms = self.surface.posed(self.nominal, (0., 0., .78), (1., 0., 0., 0.))
        obj = palms[1] + np.array([.10, 0., 0.]) + rng.normal(0, .005, (200, 3))
        wall = np.column_stack([np.full(400, .5), rng.uniform(-.5, .5, 400), rng.uniform(.1, 1.4, 400)])
        self.assertFalse(self.surface.inside(np.vstack([obj, wall]), h).any())

    def test_moving_arm_leaves_no_trail_with_point_times(self):
        q_a = self.nominal.copy(); q_b = self.nominal.copy()
        j = list(self.kin.joint_names if hasattr(self.kin, 'joint_names') else [])
        from wholebody_core import JOINT_NAMES
        i = JOINT_NAMES.index('right_shoulder_pitch_joint'); q_b[i] -= 1.0          # the arm swings forward 1 rad
        arm = [n for n in self.names if n.startswith('right_') and ('elbow' in n or 'wrist' in n or 'hand' in n)]
        old = self.surface_points(q_a, bodies=arm)                                  # seen 0.4 s ago
        h = self.history([q_a, q_a, q_a, q_a, q_b])
        timed = np.column_stack([old, np.full(len(old), h['t'][0])])
        self.assertGreater(self.surface.inside(timed, h).mean(), .97)             # posed at the scan time: removed
        self.assertLess(self.surface.inside(old, h).mean(), .2)                    # end-of-window pose: a trail remains

    def test_grip_radius_removes_held_object(self):
        _, palms = self.surface.posed(self.nominal, (0., 0., .78), (1., 0., 0., 0.))
        held = palms[0] + np.array([.08, 0., 0.]) + np.random.default_rng(1).normal(0, .015, (100, 3))   # a box held in front
        h = self.history([self.nominal])
        self.assertLess(self.surface.inside(held, h).mean(), .5)                    # default: the object is an obstacle
        near = held[np.linalg.norm(held - palms[0], axis=1) < .14]
        self.assertTrue(self.surface.inside(near, h, grip_radius=.15).all())       # grip radius: part of the robot

    def test_heading_across_180_degrees_is_interpolated_the_short_way(self):
        from g1_wholebody.frames import matrix_quat, yaw_matrix
        quat = lambda deg: matrix_quat(yaw_matrix(np.radians(deg)))         # w >= 0: 179 and -179 deg differ in sign
        h = dict(t=np.array([100., 100.1]), q=np.array([self.nominal, self.nominal]), pos=np.tile([0., 0., .78], (2, 1)),
                 quat=np.array([quat(179.), quat(-179.)]))
        self.assertLess(h['quat'][0] @ h['quat'][1], 0)
        own = self.surface_points(self.nominal, quat=quat(180.))            # the body half-way between the snapshots
        timed = np.column_stack([own, np.full(len(own), 100.05)])
        self.assertGreater(self.surface.inside(timed, h).mean(), .97)       # (the unaligned blend: heading 0, ~15 % missed)

    def test_remove_body_keeps_time_column_and_rope(self):
        h = self.history([self.nominal]); h['rope'] = np.array([0., 0., .78])
        pts = np.array([[0., 0., 1.6, 5.], [1., 1., .5, 6.]])                       # rope above the pelvis, a box
        out = L.remove_body(pts, h)
        self.assertEqual(out.shape, (1, 4)); self.assertEqual(out[0, 3], 6.)


if __name__ == '__main__':
    unittest.main()
