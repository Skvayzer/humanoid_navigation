import time
import unittest

import numpy as np

from g1_wholebody.kinematics import FastKinematics
from wholebody_core import WholeBodyKinematics, load_contract


class KinematicsTest(unittest.TestCase):
    def test_identical_to_training_kinematics(self):
        c = load_contract(); ref = WholeBodyKinematics(c); fast = FastKinematics(c)
        rng = np.random.default_rng(0)
        for _ in range(30):
            q = rng.uniform(c['soft_lower'], c['soft_upper'])
            b0, s0, j0 = ref.frames(q); b1, s1, j1 = fast.frames(q)
            self.assertEqual(set(b0), set(b1)); self.assertEqual(set(s0), set(s1)); self.assertEqual(set(j0), set(j1))
            for k in b0: np.testing.assert_allclose(b1[k], b0[k], atol=1e-12)
            for k in s0: np.testing.assert_allclose(s1[k], s0[k], atol=1e-12)
            quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
            np.testing.assert_allclose(fast.world_sites(q, [1, 2, 3], quat), ref.world_sites(q, [1, 2, 3], quat), atol=1e-12)
            np.testing.assert_allclose(fast.upper_gravity(q, quat), ref.upper_gravity(q, quat), atol=1e-9)

    def test_cache_is_per_joint_vector(self):
        c = load_contract(); fast = FastKinematics(c)
        a = c['nominal'].copy(); b = a.copy(); b[13] += 0.3
        ta = fast.frames(a)[0]['torso_link'].copy(); tb = fast.frames(b)[0]['torso_link']
        self.assertFalse(np.allclose(ta, tb))
        np.testing.assert_allclose(fast.frames(a)[0]['torso_link'], ta)

    def test_fast_enough(self):
        c = load_contract(); fast = FastKinematics(c); rng = np.random.default_rng(1)
        qs = rng.uniform(c['soft_lower'], c['soft_upper'], (200, 29))
        t = time.perf_counter()
        for q in qs:
            fast.frames(q)
        per_call = (time.perf_counter() - t) / len(qs)
        self.assertLess(per_call, 2e-3, 'FK %.2f ms' % (per_call * 1e3))


if __name__ == '__main__':
    unittest.main()
