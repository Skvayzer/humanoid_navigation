import unittest
from pathlib import Path

import numpy as np

from g1_wholebody import frames as fr
from wholebody_core import WholeBodyKinematics, load_contract, quat_matrix, WHOLEBODY


def random_quat(rng):
    q = rng.normal(size=4); return q / np.linalg.norm(q)


class FramesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract(); cls.kin = WholeBodyKinematics(cls.contract)

    def test_quaternion_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            q = random_quat(rng); r = quat_matrix(q)
            np.testing.assert_allclose(quat_matrix(fr.matrix_quat(r)), r, atol=1e-12)

    def test_chain_inverts_against_mujoco(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest('MuJoCo not installed')
        xml = (WHOLEBODY / 'robot.xml').read_text()
        import re
        xml = re.sub(r'<mesh [^>]*/>', '', xml); xml = re.sub(r'<geom [^>]*mesh="[^"]*"[^>]*/>', '', xml)
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        site = model.site('mid360_site').id
        chain = fr.PelvisChain(self.kin, 'robot_xml')
        rng = np.random.default_rng(1)
        lo, hi = self.contract['soft_lower'], self.contract['soft_upper']
        for _ in range(25):
            data.qpos[:3] = rng.uniform(-3, 3, 3); data.qpos[3:7] = random_quat(rng)
            data.qpos[7:] = rng.uniform(lo, hi)
            mujoco.mj_forward(model, data)
            t_map_lidar = fr.transform(data.site_xmat[site].reshape(3, 3), data.site_xpos[site])
            t_map_body = t_map_lidar @ fr.transform(np.eye(3), -fr.LIDAR_TO_IMU_T)
            t = chain.pelvis_from_body(t_map_body, data.qpos[7:].copy())
            np.testing.assert_allclose(t[:3, 3], data.qpos[:3], atol=1e-9)
            np.testing.assert_allclose(t[:3, :3], quat_matrix(data.qpos[3:7]), atol=1e-9)
            back = chain.body_from_pelvis(fr.pose_matrix(data.qpos[:3], data.qpos[3:7]), data.qpos[7:].copy())
            np.testing.assert_allclose(back, t_map_body, atol=1e-9)

    def test_fusion_keeps_imu_tilt_and_chain_heading(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            true = quat_matrix(random_quat(rng))
            if abs(true[2, 2]) < 0.3:
                continue                       # heading undefined near 90 deg tilt (not a walking pose)
            imu = fr.yaw_matrix(rng.uniform(-np.pi, np.pi)) @ true       # IMU yaw drifts arbitrarily
            chain = fr.yaw_matrix(0.0) @ true @ fr.rpy_matrix(*rng.normal(0, 0.05, 3))   # chain tilt slightly off
            fused = fr.fuse_orientation(chain, imu)
            np.testing.assert_allclose(fused[2], true[2], atol=1e-9)                       # gravity from the IMU
            self.assertAlmostEqual(fr.heading(fused), fr.heading(chain), places=9)

    def test_mounts_differ_by_urdf_pitch(self):
        a = fr.PelvisChain(self.kin, 'robot_xml').t_torso_body; b = fr.PelvisChain(self.kin, 'unitree_urdf').t_torso_body
        angle = np.degrees(np.arccos((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2))
        self.assertAlmostEqual(angle, 2.929, places=2)


if __name__ == '__main__':
    unittest.main()
