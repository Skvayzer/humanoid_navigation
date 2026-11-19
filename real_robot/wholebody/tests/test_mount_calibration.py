import unittest

import numpy as np

from g1_wholebody.frames import MountCalibrator, PelvisChain, matrix_quat, rpy_matrix, transform
from g1_wholebody.kinematics import FastKinematics
from wholebody_core import load_contract


class MountCalibrationTest(unittest.TestCase):
    """The head (LiDAR) is not where the URDF says: the calibrator finds its roll/pitch from standing samples."""

    def test_converges_to_the_true_mount_and_fixes_the_pelvis_position(self):
        c = load_contract(); kin = FastKinematics(c); q = np.asarray(c['nominal'], float)
        true = PelvisChain(kin); true.set_mount_correction(np.radians(2.), np.radians(11.5))
        chain = PelvisChain(kin); cal = MountCalibrator(chain)
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(40):                                 # standing: small sway, any heading
            r_p = rpy_matrix(*np.radians(rng.normal(0, 1.5, 2)), rng.uniform(-np.pi, np.pi))
            t_map_pelvis = transform(r_p, [1., 2., .78])
            t_map_body = true.body_from_pelvis(t_map_pelvis, q)
            samples.append((t_map_body, matrix_quat(r_p), t_map_pelvis))
        self.assertIsNone(cal.warning())
        for _ in range(12):                                 # 12 s of standing, 1 Hz updates
            for t_map_body, imu, _ in samples[:10]:
                cal.add(t_map_body, q, imu)
            cal.update()
        roll, pitch = np.degrees(chain.correction)
        self.assertAlmostEqual(pitch, 11.5, delta=.2); self.assertAlmostEqual(roll, 2., delta=.2)
        self.assertIn('pitch +11.', cal.warning())
        t_map_body, imu, t_map_pelvis = samples[-1]
        pos, _, _ = chain.pelvis_pose(t_map_body, q, imu)
        self.assertLess(np.linalg.norm(pos - t_map_pelvis[:3, 3]), .01)
        uncorrected, _, _ = PelvisChain(kin).pelvis_pose(t_map_body, q, imu)
        self.assertGreater(np.linalg.norm(uncorrected - t_map_pelvis[:3, 3]), .05)   # the error it removes

    def test_solve_at_start_from_one_second_of_still_samples(self):
        c = load_contract(); kin = FastKinematics(c); q = np.asarray(c['nominal'], float)
        true = PelvisChain(kin); true.set_mount_correction(np.radians(-1.5), np.radians(12.4))
        chain = PelvisChain(kin); rng = np.random.default_rng(1); samples = []
        for _ in range(20):                                     # hanging in the gantry, tilted, hardly moving
            r_p = rpy_matrix(np.radians(3 + rng.normal(0, .2)), np.radians(-6 + rng.normal(0, .2)), 2.1)
            t_map_pelvis = transform(r_p, [0.3, -1.2, .95])
            samples.append((true.body_from_pelvis(t_map_pelvis, q), q, matrix_quat(r_p)))
        self.assertTrue(MountCalibrator(chain).solve(samples))
        np.testing.assert_allclose(np.degrees(chain.correction), [-1.5, 12.4], atol=.2)
        pos, _, _ = chain.pelvis_pose(samples[0][0], q, samples[0][2])
        self.assertLess(np.linalg.norm(pos - [0.3, -1.2, .95]), .01)        # the scene anchor is at the robot
        fresh = PelvisChain(kin)
        self.assertFalse(MountCalibrator(fresh).solve(samples[:3])); self.assertEqual(fresh.correction, (0., 0.))

    def test_too_few_samples_change_nothing(self):
        c = load_contract(); chain = PelvisChain(FastKinematics(c)); cal = MountCalibrator(chain)
        cal.add(np.eye(4), np.asarray(c['nominal'], float), [1., 0., 0., 0.])
        self.assertFalse(cal.update()); self.assertEqual(chain.correction, (0., 0.))


if __name__ == '__main__':
    unittest.main()
