import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as Obj

import numpy as np

from checkpoint import verify
from field_core import compute_fields, sample_fields
from perception_core import SHAPE, RESOLUTION
from state_core import Kinematics, StateBuffer, JOINT_NAMES, telemetry


def message(tick=1):
    return Obj(tick=tick, mode_pr=0, mode_machine=0,
               motor_state=[Obj(q=0., dq=0., tau_est=0.) for _ in range(35)],
               imu_state=Obj(quaternion=[1., 0., 0., 0.], gyroscope=[0., 0., 0.],
                             accelerometer=[0., 0., 9.81]))


class TelemetryTests(unittest.TestCase):
    def test_identity_gravity_and_joint_order(self):
        value = telemetry(message())
        np.testing.assert_allclose(value['gravity'], [0, 0, -1])
        self.assertEqual(len(JOINT_NAMES), 29)
        self.assertEqual(JOINT_NAMES[12:15], ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'))

    def test_duplicate_does_not_refresh_receive_time(self):
        buf = StateBuffer()
        buf.receive(message(), 10.)
        self.assertFalse(buf.receive(message(), 10.24))
        with self.assertRaisesRegex(ValueError, 'stale'):
            buf.snapshot(10.3)

    def test_invalid_telemetry_clears_last_valid_value(self):
        for mutate in (lambda m: setattr(m, 'mode_pr', 1),
                       lambda m: setattr(m.motor_state[0], 'q', float('nan')),
                       lambda m: setattr(m.imu_state, 'quaternion', [0]*4)):
            buf = StateBuffer()
            buf.receive(message(), 1.)
            bad = message(2)
            mutate(bad)
            with self.assertRaises(ValueError):
                buf.receive(bad, 1.1)
            with self.assertRaises(ValueError):
                buf.snapshot(1.1)

    def test_counter_wrap_is_valid_but_backwards_is_not(self):
        buf = StateBuffer()
        buf.receive(message(0xfffffffe), 10.)
        self.assertTrue(buf.receive(message(1), 10.1))
        with self.assertRaisesRegex(ValueError, 'backwards'):
            buf.receive(message(0), 10.2)

    def test_future_receive_time_is_not_fresh(self):
        buf = StateBuffer()
        buf.receive(message(), 10.)
        with self.assertRaises(ValueError):
            buf.snapshot(9.)


class KinematicsTests(unittest.TestCase):
    def test_joint_mapping_and_model_hash(self):
        model = Path(__file__).resolve().parents[1]/'upstream/g1_mjx_feetonly_torque.xml'
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(),
                         '7e2b294fc335d388613000ff333e6b32aa81052b0fba41ee3bc311dfacc43499')
        self.assertEqual(Kinematics().limits.shape, (29, 2))

    def test_waist_moves_upper_body_not_feet(self):
        k = Kinematics()
        q = np.zeros(29)
        original = k.sites(q)
        q[13] = .2
        moved = k.sites(q)
        np.testing.assert_allclose(original['left_foot'], moved['left_foot'])
        self.assertGreater(np.linalg.norm(original['head']-moved['head']), .1)

    def test_out_of_model_limits_rejected(self):
        with self.assertRaises(ValueError):
            Kinematics().sites(np.full(29, 100.))

    def test_against_mujoco_reference_fixtures(self):
        fixture = json.loads((Path(__file__).parent/'fk_reference.json').read_text())
        k = Kinematics()
        for case in fixture['cases']:
            actual = k.sites(case['q'])
            for name, matrix in case['sites'].items():
                np.testing.assert_allclose(actual[name], matrix, atol=2e-7)


class FieldTests(unittest.TestCase):
    def grid(self):
        state = np.ones(SHAPE, np.uint8)
        state[:20] = 2
        return state

    def test_wall_distance_normal_and_no_automatic_goal(self):
        state = self.grid()
        fields = compute_fields(state, state == 2, [0, 0, 0])
        self.assertGreater(fields.sdf[25, 64, 15], 0)
        self.assertLess(fields.sdf[15, 64, 15], 0)
        self.assertGreater(fields.boundary[25, 64, 15, 0], .9)
        self.assertEqual(np.count_nonzero(fields.guidance), 0)

    def test_numerical_parity_with_pinned_upstream_in_known_space(self):
        state = self.grid()
        fields = compute_fields(state, state == 2, [0, 0, 0], [3.5, 2.5, .75])
        fixture = json.loads((Path(__file__).parent/'field_reference.json').read_text())
        for sample in fixture['samples']:
            index = tuple(sample['index'])
            self.assertAlmostEqual(float(fields.sdf[index]), sample['sdf'], places=5)
            np.testing.assert_allclose(fields.boundary[index], sample['boundary'], atol=1e-5)
            np.testing.assert_allclose(fields.guidance[index], sample['guidance'], atol=1e-5)

    def test_goal_guidance_points_forward(self):
        state = self.grid()
        fields = compute_fields(state, state == 2, [0, 0, 0], [3.5, 2.5, .75])
        self.assertGreater(fields.guidance[60, 62, 18, 0], .4)
        self.assertTrue(fields.reachable[60, 62, 18])

    def test_unknown_goal_and_outside_goal_rejected(self):
        state = self.grid()
        state[80:100] = 0
        for goal in ([3.5, 2.5, .75], [8., 0., .75]):
            with self.assertRaises(ValueError):
                compute_fields(state, state == 2, [0, 0, 0], goal)

    def test_unreachable_side_has_no_arrows(self):
        state = self.grid()
        state[64:70] = 0
        fields = compute_fields(state, state == 2, [0, 0, 0], [4., 2.5, .75])
        self.assertFalse(fields.reachable[40, 62, 18])
        np.testing.assert_array_equal(fields.guidance[40, 62, 18], 0)

    def test_empty_known_scene_without_fake_obstacles(self):
        state = np.ones(SHAPE, np.uint8)
        fields = compute_fields(state, np.zeros(SHAPE, bool), [0, 0, 0])
        self.assertTrue((fields.sdf > 0).all())
        self.assertEqual(np.count_nonzero(fields.boundary), 0)

    def test_all_unknown_is_not_empty_free_scene(self):
        with self.assertRaisesRegex(ValueError, 'free space'):
            compute_fields(np.zeros(SHAPE, np.uint8), np.zeros(SHAPE, bool), [0, 0, 0])

    def test_thin_raw_obstacle_survives_morphology_loss(self):
        state = self.grid()
        state[70, 70, 18] = 2
        fields = compute_fields(state, np.zeros(SHAPE, bool), [0, 0, 0])
        self.assertFalse(fields.known_free[70, 70, 18])

    def test_sampling_is_voxel_centered_and_rejects_unknown_corner(self):
        state = self.grid()
        fields = compute_fields(state, state == 2, [0, 0, 0])
        pos = (np.array([30, 50, 20])+.5)*RESOLUTION
        self.assertAlmostEqual(sample_fields(fields, [pos])['sdf'][0], fields.sdf[30, 50, 20])
        with self.assertRaises(ValueError):
            sample_fields(fields, [[-.01, 0, 0]])
        with self.assertRaises(ValueError):
            sample_fields(fields, [[20*RESOLUTION, 1., .5]])


class ArtifactTests(unittest.TestCase):
    def test_modified_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory(prefix='cat-artifact-') as directory:
            (Path(directory)/'policy.onnx').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'hash/size'):
                verify(directory)


class ResearchSafetyTests(unittest.TestCase):
    def test_runtime_only_publishes_visualization_and_never_loads_policy(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ('research_node.py', 'field_core.py', 'state_core.py'):
            source = (root/filename).read_text()
            tree = ast.parse(source)
            for forbidden in ('LowCmd', 'ChannelPublisher', 'TransformBroadcaster', 'create_client',
                              'ActionClient', 'onnxruntime', 'unitree_sdk', 'shared_memory', 'subprocess'):
                self.assertNotIn(forbidden, source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'create_publisher':
                    self.assertIsInstance(node.args[1], ast.Constant)
                    self.assertTrue(node.args[1].value.startswith('/g1_cat/'))


if __name__ == '__main__':
    unittest.main()
