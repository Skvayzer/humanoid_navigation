import unittest

import numpy as np

from g1_wholebody.ros_executor import StampAge


class PoseAgeTest(unittest.TestCase):
    """The SLAM pose's age when its stamps are on another clock (the vendor board's, ~20 min off the Jetson's)."""

    def test_age_is_the_latency_whatever_the_clock_offset(self):
        rng = np.random.default_rng(0)
        for offset in (-1302.4, 0., 977.):                      # stamp clock minus ours; "now - stamp" gave 0.5 s or 0
            age = StampAge(window=250, floor_s=0.01); got, want = [], []
            latency = rng.uniform(0.012, 0.06, 400)
            for k, lat in enumerate(latency):
                t = 0.02 * k                                    # when the pose was taken, on our (monotonic) clock
                got.append(age(t + lat, t + offset))
                want.append(lat - latency[max(0, k - 249):k + 1].min() + 0.01)   # the shortest latency of the window
            np.testing.assert_allclose(got, want, atol=1e-6)    # (stamps of ~1e3 s: float rounding of ~1e-7)
            self.assertLess(max(got[50:]), 0.07)                # the latency (< 60 ms), never the 0.5 s clip

    def test_bounds_and_window(self):
        age = StampAge(window=10, floor_s=0.02, max_s=0.5)
        self.assertEqual(age(5.0, 4.0), 0.02)                   # the first message: only the floor is known
        self.assertEqual(age(9.0, 4.0), 0.5)                    # clipped
        for k in range(10):                                     # the window forgets: a stepped stamp clock recovers
            a = age(10. + k, 14. + k)
        self.assertAlmostEqual(a, 0.02)


if __name__ == '__main__':
    unittest.main()
