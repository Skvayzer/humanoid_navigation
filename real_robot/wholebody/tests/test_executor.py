import unittest

import numpy as np

from g1_wholebody import runner as rn
from g1_wholebody.fields import FieldPlanner, VirtualScene, solve_fields
from g1_wholebody.frames import PelvisChain, pose_matrix
from g1_wholebody.kinematics import FastKinematics
from g1_wholebody.lowlevel import LowState, remote_bytes
from wholebody_core import load_contract


class Clock:
    def __init__(self): self.t = 0.
    def __call__(self): return self.t


class InstantFields:
    """Pre-solved fields (the solve itself must never run inside a control step)."""
    cache = {}

    def __init__(self, scene):
        self.scene, self.out, self.requests, self.busy = scene, None, 0, False
        key = scene.name
        if key not in self.cache:
            self.cache[key] = solve_fields(scene.start, scene.waypoints()[0], scene.occupancy)

    def request(self, xy, goal):
        if self.out is not None or self.busy:
            return False
        self.out = (self.cache[self.scene.name][0], np.asarray(goal, float)); self.requests += 1; return True

    def poll(self):
        out, self.out = self.out, None
        return out


class ExecutorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_contract(); cls.kin = FastKinematics(cls.c)

    def make(self, allow_active=True, geofence=(-10, 10, -10, 10), infer=None):
        self.clock = Clock(); self.scene = VirtualScene('sidegap_plain_047', anchor=(0., 0., 0.))
        self.fields = InstantFields(self.scene)
        # safety-logic tests must not depend on how fast this machine is (QEMU is ~10x slower):
        # the step-time watchdog has its own test below
        cfg = rn.Config(allow_active=allow_active, geofence=geofence,
                        limits=rn.Limits(step_budget_s=10., step_hard_limit_s=10.))
        self.calls = []
        def default_infer(obs):
            self.calls.append(obs); return np.zeros(29)
        ex = rn.Executor(self.c, self.kin, infer or default_infer, FieldPlanner(self.scene), self.fields, cfg, clock=self.clock)
        self.chain = PelvisChain(self.kin)
        return ex

    def feed(self, ex, buttons=(), q=None, pelvis=None, pose=True):
        q = self.c['nominal'].copy() if q is None else q
        pos = np.r_[self.scene.start, .78] if pelvis is None else pelvis
        ex.on_lowstate(LowState(tick=int(self.clock.t * 500), mode_pr=0, mode_machine=5, q=q, dq=np.zeros(29), tau=np.zeros(29),
                                motor_mode=np.ones(29), temperature=np.full(29, 40), quat_wxyz=np.array([1., 0, 0, 0]),
                                gyro=np.zeros(3), acc=np.array([0, 0, 9.81]), rpy=np.zeros(3), remote=remote_bytes(*buttons),
                                received=self.clock.t))
        if pose:
            ex.on_slam_pose(self.chain.body_from_pelvis(pose_matrix(pos, [1, 0, 0, 0]), q), self.clock.t, received=self.clock.t)

    def run_steps(self, ex, n, **feed):
        out = []
        for _ in range(n):
            self.clock.t += 0.02; self.feed(ex, **feed); out.append(ex.step())
        return out

    def to_policy(self, ex):
        self.run_steps(ex, 2)
        self.assertTrue(ex.command('stand'), ex.reason)
        self.run_steps(ex, int(3.1 / 0.02))
        self.assertEqual(ex.mode, rn.STAND)
        self.assertTrue(ex.command('go'), ex.reason)
        cmds = self.run_steps(ex, 3)
        self.assertEqual(ex.mode, rn.POLICY, ex.reason)
        return cmds

    # ------------------------------------------------------------------
    def test_idle_and_shadow_never_command(self):
        ex = self.make()
        self.assertTrue(all(c is None for c in self.run_steps(ex, 10)))
        self.assertTrue(ex.command('shadow'))
        cmds = self.run_steps(ex, 50)
        self.assertTrue(all(c is None for c in cmds))
        self.assertGreater(len(self.calls), 40)             # the policy did run on live data

    def test_active_requires_flag(self):
        ex = self.make(allow_active=False)
        self.run_steps(ex, 2)
        self.assertFalse(ex.command('stand')); self.assertIn('--active', ex.reason)
        self.assertEqual(ex.mode, rn.IDLE)

    def test_policy_requires_geofence(self):
        ex = self.make(geofence=None)
        self.run_steps(ex, 2); ex.command('stand'); self.run_steps(ex, 160)
        self.assertFalse(ex.command('go')); self.assertIn('geofence', ex.reason)

    def test_stand_up_ramps_gains_and_pose(self):
        ex = self.make(); self.run_steps(ex, 2)
        hanging = self.c['nominal'] + 0.4
        self.feed(ex, q=hanging); ex.command('stand')
        first = self.run_steps(ex, 1, q=hanging)[0]
        self.assertLess(first.kp[3], 0.3 * self.c['kp'][3])
        np.testing.assert_allclose(first.q[:29], hanging, atol=0.02)
        last = self.run_steps(ex, 160)[-1]
        np.testing.assert_allclose(last.q[:29], self.c['nominal'], atol=1e-6)
        np.testing.assert_allclose(last.kp[:29], self.c['kp'], rtol=1e-6)

    def test_policy_commands_contract_gains_and_feedforward(self):
        ex = self.make(); cmd = self.to_policy(ex)[-1]
        np.testing.assert_allclose(cmd.kp[:29], self.c['kp']); np.testing.assert_allclose(cmd.kd[:29], self.c['kd'])
        self.assertTrue(np.all(cmd.tau[:12] == 0) and np.any(cmd.tau[12:29] != 0))
        self.assertTrue((cmd.mode[:29] == 1).all())

    def test_balance_waits_for_fields_of_the_hold_point(self):
        ex = self.make(); self.run_steps(ex, 2)
        self.assertTrue(ex.command('stand'), ex.reason); self.run_steps(ex, int(3.1 / 0.02))
        # the worker is still solving an older question; the fields at hand lead to a waypoint 1.6 m away
        ex.field_goal, ex.field_hold = np.asarray(self.scene.waypoints()[1], float), False; self.fields.busy = True
        self.assertTrue(ex.command('balance'), ex.reason); self.calls.clear()
        cmd = self.run_steps(ex, 5)[-1]
        self.assertEqual(ex.mode, rn.POLICY); self.assertEqual(self.calls, [])
        self.assertEqual(ex.last['policy_wait'], 'fields of the hold point')
        np.testing.assert_allclose(cmd.q[:29], self.c['nominal'], atol=1e-6)        # holds the stand pose
        self.fields.busy = False; ex.next_field_request = 0.; self.run_steps(ex, 3)
        self.assertTrue(self.calls); self.assertIsNone(ex.last['policy_wait'])

    def test_new_goal_while_stopping_walks_on(self):
        ex = self.make(); self.to_policy(ex)
        ex.builder.stop = 30; phase = ex.builder.phase.copy()      # stopping on the old goal's fields
        ex._take_fields(self.clock.t, ex.fields, np.asarray(ex.field_goal) + 1.)
        self.assertEqual(ex.builder.stop, 100); np.testing.assert_array_equal(ex.builder.phase, phase)
        ex.builder.stop = 0                                         # stopped for good: restarted
        ex._take_fields(self.clock.t, ex.fields, np.asarray(ex.field_goal) + 1.)
        self.assertEqual(ex.builder.stop, 100)

    def assert_damped(self, cmd, reason_part, ex):
        self.assertEqual(ex.mode, rn.DAMP); self.assertIn(reason_part, ex.reason)
        self.assertTrue(np.all(cmd.kp == 0) and np.all(cmd.kd[:29] > 0) and np.all(cmd.tau == 0))

    def test_remote_estop(self):
        ex = self.make(); self.to_policy(ex)
        cmd = self.run_steps(ex, 1, buttons=('L2', 'B'))[0]
        self.assert_damped(cmd, 'operator', ex)

    def test_lowstate_dropout(self):
        ex = self.make(); self.to_policy(ex)
        for _ in range(3):
            self.clock.t += 0.02; cmd = ex.step()           # no new LowState for 60 ms
        self.assert_damped(cmd, 'LowState stale', ex)

    def test_pose_freeze(self):
        ex = self.make(); self.to_policy(ex)
        cmd = None
        for _ in range(20):
            self.clock.t += 0.02; self.feed(ex, pose=False); cmd = ex.step()
        self.assert_damped(cmd, 'SLAM pose stale', ex)

    def test_pose_jump(self):
        ex = self.make(); self.to_policy(ex)
        cmd = self.run_steps(ex, 1, pelvis=np.r_[self.scene.start + [0.6, 0.], .78])[0]
        self.assert_damped(cmd, 'jump', ex)

    def test_geofence(self):
        x, y = VirtualScene('sidegap_plain_047', anchor=(0., 0., 0.)).start
        ex = self.make(geofence=(x - .2, x + .2, y - .2, y + .2)); self.to_policy(ex)
        cmds = []
        for k in range(1, 30):    # walk the reported pose out slowly (no jump)
            cmds += self.run_steps(ex, 1, pelvis=np.r_[x + 0.02 * k, y, .78])
            if ex.mode == rn.DAMP:
                break
        self.assert_damped(cmds[-1], 'geofence', ex)

    def test_damp_is_latched_and_keeps_publishing(self):
        ex = self.make(); self.to_policy(ex)
        ex.command('damp')
        self.assertFalse(ex.command('go')); self.assertFalse(ex.command('stand')); self.assertFalse(ex.command('idle'))
        cmds = self.run_steps(ex, 5)
        self.assertTrue(all(c is not None and np.all(c.kp == 0) for c in cmds))
        self.assertTrue(ex.command('reset')); self.assertEqual(ex.mode, rn.IDLE)
        self.assertTrue(all(c is None for c in self.run_steps(ex, 3)))

    def test_bad_policy_output_damps(self):
        ex = self.make(infer=lambda obs: np.full(29, np.nan)); self.run_steps(ex, 2)
        ex.command('stand'); self.run_steps(ex, 160); ex.command('go')
        cmd = self.run_steps(ex, 1)[0]
        self.assert_damped(cmd, 'invalid policy output', ex)

    def test_step_overrun_damps(self):
        import time as _time
        slow = {'on': False}
        def infer(obs):
            if slow['on']:
                _time.sleep(0.06)                       # a policy call that blows the 40 ms hard limit
            return np.zeros(29)
        ex = self.make(infer=infer); ex.cfg.limits = rn.Limits()      # default timing limits
        self.run_steps(ex, 2); ex.command('stand'); self.run_steps(ex, 160); ex.command('go'); self.run_steps(ex, 2)
        slow['on'] = True
        self.run_steps(ex, 1)
        self.assertEqual(ex.mode, rn.DAMP); self.assertIn('overrun', ex.reason)
        cmd = self.run_steps(ex, 1)[0]
        self.assertTrue(np.all(cmd.kp == 0))

    def test_remote_hold_required_for_stand(self):
        ex = self.make(); self.run_steps(ex, 2)
        self.run_steps(ex, 10, buttons=('L1', 'A'))         # 0.2 s: not yet
        self.assertEqual(ex.mode, rn.IDLE)
        self.run_steps(ex, 20, buttons=('L1', 'A'))         # > 0.5 s held
        self.assertIn(ex.mode, (rn.STAND_UP, rn.STAND))


    def test_remote_drives_the_whole_cycle(self):
        # DAMP -> IDLE -> STAND -> POLICY (hold) -> POLICY (walking), each combo held 0.6 s, no operator topic
        ex = self.make(); self.run_steps(ex, 2)
        self.run_steps(ex, 2, buttons=('L2', 'B')); self.assertEqual(ex.mode, rn.DAMP)
        self.run_steps(ex, 2); self.run_steps(ex, 30, buttons=('L1', 'Y')); self.assertEqual(ex.mode, rn.IDLE)
        self.run_steps(ex, 2); self.run_steps(ex, 30, buttons=('L1', 'A')); self.run_steps(ex, 160)
        self.assertEqual(ex.mode, rn.STAND)
        self.run_steps(ex, 30, buttons=('L1', 'X')); self.assertEqual(ex.mode, rn.POLICY)
        self.assertIsNotNone(ex.planner.hold_xy)
        self.run_steps(ex, 5); self.run_steps(ex, 30, buttons=('R1', 'A'))
        self.assertEqual(ex.mode, rn.POLICY); self.assertIsNone(ex.planner.hold_xy)

    def test_operator_goal_and_world_changes(self):
        ex = self.make(geofence=(-1, 3, -1, 1)); self.to_policy(ex)
        self.assertFalse(ex.set_goal((2.9, 0.0)))                      # 0.1 m from the fence: refused
        self.assertTrue(ex.set_goal((2.0, 0.5)))
        np.testing.assert_allclose(ex.planner.waypoints[-1], [2.0, 0.5])
        self.assertFalse(ex.replace_world(FieldPlanner(self.scene)))   # walking: refused
        self.assertTrue(ex.command('stop')); self.assertTrue(ex.replace_world(FieldPlanner(self.scene)))
        self.assertIsNotNone(ex.planner.hold_xy)                       # still holding after the change


if __name__ == '__main__':
    unittest.main()
