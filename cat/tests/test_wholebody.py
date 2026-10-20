"""Whole-body policy runtime vs recorded training-simulator steps (tests/wholebody_reference.npz).

The fixture holds 320 deterministic steps of the training task (CPU MuJoCo) in two scenes, the raw state each
observation was built from, the policy's actions, and the scene fields. The runtime must rebuild every one of the
226 observation entries from the raw state alone.
"""
from pathlib import Path
import unittest

import numpy as np

from wholebody_core import (FieldGrid, ObservationBuilder, WholeBodyKinematics, load_contract, motor_targets,
                            quat_matrix)

FIXTURE = Path(__file__).with_name('wholebody_reference.npz')


def load():
    data = np.load(FIXTURE)
    grids = {}
    for n in range(len(data['scene_names'])):
        grids[n] = FieldGrid(data['grid%d_sdf' % n], data['grid%d_gf' % n], data['grid%d_bf' % n],
                             data['grid%d_origin' % n], float(data['grid%d_dx' % n]))
    return data, grids


class WholeBodyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract()
        cls.kin = WholeBodyKinematics(cls.contract)
        cls.data, cls.grids = load()

    def test_kinematics_matches_simulator_sites(self):
        d = self.data
        worst = 0.
        for k in range(0, len(d['obs']), 7):
            qpos = d['qpos'][k]
            sites = self.kin.world_sites(qpos[7:], qpos[:3], qpos[3:7])
            expected = d['sites_final'][k]
            worst = max(worst, float(np.abs(sites-expected).max()))
        self.assertLess(worst, 2e-5)

    def test_gravity_feedforward_matches_mujoco(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest('MuJoCo not installed (development check only)')
        import re
        xml = (Path(__file__).resolve().parents[1]/'wholebody'/'robot.xml').read_text()
        xml = re.sub(r'<asset>.*?</asset>', '', xml, flags=re.S)
        xml = re.sub(r'<geom[^>]*\smesh="[^"]*"[^>]*/>', '', xml)
        xml = re.sub(r'<pair[^>]*/>', '', xml)
        xml = re.sub(r'\smaterial="[^"]*"', '', xml)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        rng = np.random.default_rng(0)
        for _ in range(5):
            q = self.contract['nominal']+rng.uniform(-.3, .3, 29)
            quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
            data.qpos[:] = 0; data.qpos[3:7] = quat; data.qpos[7:] = q; data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            expected = data.qfrc_bias[6+12:]                    # zero velocity: bias = gravity generalised force
            got = self.kin.upper_gravity(q, quat)
            np.testing.assert_allclose(got, expected, atol=2e-3, rtol=1e-3)

    def test_field_sampler_reproduces_training_interpolation(self):
        grid = self.grids[0]
        rng = np.random.default_rng(1)
        points = grid.origin+rng.uniform(1, 20, (50, 3))*grid.dx
        gf, bf, sdf = grid.sample(points)
        # Independent re-implementation of the training quirk: weight index (x,y,z) with z fastest paired with the
        # x-fast corner list, i.e. the fractional parts of x and z exchange roles in the weights.
        xyz = (points-grid.origin)/grid.dx; base = np.floor(xyz).astype(int); f = xyz-base
        corners = FieldGrid.CORNERS
        expected = np.zeros(len(points))
        for k, (a, b, c) in enumerate(corners):
            wa, wb, wc = (k >> 2) & 1, (k >> 1) & 1, k & 1           # z-fast weight order
            w = np.where(wa, f[:, 0], 1-f[:, 0])*np.where(wb, f[:, 1], 1-f[:, 1])*np.where(wc, f[:, 2], 1-f[:, 2])
            expected += w*grid.sdf[base[:, 0]+a, base[:, 1]+b, base[:, 2]+c]
        np.testing.assert_allclose(sdf, expected, atol=1e-5)

    def test_observations_match_training(self):
        d = self.data
        builder = ObservationBuilder(self.contract, self.kin)
        worst, worst_feature, compared = 0., None, 0
        names = self.contract['feature_names']
        start = True
        for k in range(len(d['obs'])):
            qpos, qvel, grid = d['qpos'][k], d['qvel'][k], self.grids[int(d['scene'][k])]
            # Frequency is drawn per episode in training; recover it from the recorded phase advance.
            # Training sampled fields at site poses one physics substep behind qpos (MuJoCo's post-step kinematics):
            # inject exactly those for the replay; on the robot the runtime uses forward kinematics at the current state.
            used = dict(sites=np.vstack((d['sites'][k], d['elbow_sites'][k])), pelvis_rotation=d['pelvis_rotation'][k])
            if start:
                obs = builder.reset(qpos[7:], qpos[:3], qpos[3:7], grid, dq=qvel[6:], gyro=d['gyro'][k],
                                    left_first=bool(d['obs'][k][names.index('phase.cos.left')] > 0), **used)
                builder.phase_dt = self._phase_dt(k)
            else:
                builder.act(d['action'][k-1])
                obs = builder.observe(qpos[7:], qvel[6:], d['gyro'][k], qpos[:3], qpos[3:7], grid, **used)
            error = np.abs(obs-d['obs'][k])
            if error.max() > worst:
                worst, worst_feature = float(error.max()), (k, names[int(error.argmax())])
            compared += 1
            start = bool(d['done'][k]) or (k+1 < len(d['obs']) and d['scene'][k+1] != d['scene'][k])
        self.assertEqual(compared, len(d['obs']))
        self.assertLess(worst, 1e-4, msg='worst %.2e at step/feature %s' % (worst, worst_feature))

    def _phase_dt(self, k):
        names = self.contract['feature_names']
        i = names.index('phase.cos.left'), names.index('phase.sin.left')
        a = np.arctan2(self.data['obs'][k][i[1]], self.data['obs'][k][i[0]])
        b = np.arctan2(self.data['obs'][k+1][i[1]], self.data['obs'][k+1][i[0]])
        return float(np.mod(b-a, 2*np.pi))

    def test_motor_target_rule(self):
        c = self.contract
        previous = c['nominal'].copy()
        targets = motor_targets(np.ones(29), previous, c)
        np.testing.assert_allclose(targets[:12], np.clip(previous[:12]+.5, c['soft_lower'][:12], c['soft_upper'][:12]))
        step = c['upper_target_rate']*c['control_dt']
        self.assertTrue(np.all(np.abs(targets[12:]-previous[12:]) <= step+1e-12))

    def test_gravity_observation_convention(self):
        rotation = quat_matrix([np.cos(.2), np.sin(.2), 0, 0])
        self.assertTrue(np.allclose(-rotation[2, :], rotation.T @ np.array([0, 0, -1.])))


if __name__ == '__main__':
    unittest.main()
