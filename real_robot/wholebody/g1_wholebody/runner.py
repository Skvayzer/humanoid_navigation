"""Whole-body policy executor: modes, watchdogs and the 50 Hz control step. No middleware in here.

The caller feeds LowState (pelvis IMU + joints), SLAM poses (T_map_body) and operator commands, and calls
step(now) every 20 ms; step returns the LowCmd to publish (or None in IDLE/SHADOW). The ROS node and the
closed-loop MuJoCo harness wrap exactly this class.

Modes
  IDLE      nothing is published (Unitree's own controller, or nothing, drives the robot)
  SHADOW    the policy runs on live data, actions are logged, NOTHING is ever sent to the motors
  STAND_UP  3 s ramp from the measured posture to the nominal pose (robot hanging on the gantry)
  STAND     PD hold of the nominal pose (gantry only: there is no balance control in this mode)
  POLICY    policy targets + PD gains of the training contract + waist/arm gravity feedforward
  DAMP      kp = 0, kd = damping on all body joints; latched, entered on any fault or the remote's L2+B

Active modes (STAND_UP..DAMP) publish every step, including DAMP: the motor drivers keep executing the last
command they received, so the executor must keep sending damping instead of going quiet.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from wholebody_core import ObservationBuilder
from .frames import PelvisChain, heading, yaw_matrix
from wholebody_core import quat_matrix
from .lowlevel import LowCmd, NUM_BODY, remote_buttons

IDLE, SHADOW, STAND_UP, STAND, POLICY, DAMP = 'IDLE', 'SHADOW', 'STAND_UP', 'STAND', 'POLICY', 'DAMP'
ACTIVE = (STAND_UP, STAND, POLICY, DAMP)


@dataclass
class Limits:
    lowstate_timeout_s: float = 0.05        # 25 LowState periods
    pose_timeout_s: float = 0.35            # SLAM publishes at 10 Hz
    pose_jump_m: float = 0.15               # + pose_speed_mps * dt between consecutive SLAM updates
    pose_jump_deg: float = 10.0             # + pose_rate_dps * dt
    pose_speed_mps: float = 1.5
    pose_rate_dps: float = 300.0
    fields_timeout_s: float = 8.0           # virtual scenes are static; the Jetson needs ~2-3 s per solve
    step_budget_s: float = 0.018            # observe + policy + command; the policy dies at a full 20 ms delay
    step_budget_violations: int = 5         # consecutive
    step_hard_limit_s: float = 0.040
    max_tilt_deg: float = 45.0              # pelvis IMU
    min_pelvis_z_m: float = 0.45            # the sim's fall criterion
    joint_margin_rad: float = 0.20          # beyond the contract's soft limits
    max_policy_s: float = 180.0
    max_motor_temp_c: int = 90


@dataclass
class Config:
    mount: str = 'unitree_urdf'
    kp_scale: float = 1.0
    damp_kd: Optional[np.ndarray] = None    # default: contract kd x 2
    stand_up_s: float = 3.0
    stand_up_kp_start: float = 0.2          # fraction of the contract kp at the start of the ramp
    geofence: Optional[tuple] = None        # (xmin, xmax, ymin, ymax) of REAL free floor in the map; required for POLICY
    use_imu_tilt: bool = True
    imu_heading: bool = True                # heading between SLAM updates from the pelvis IMU (offset fixed at the scan time)
    allow_active: bool = False              # the node sets this only with --active; otherwise motors are never driven
    remote_estop: frozenset = frozenset({'L2', 'B'})
    remote_stand: frozenset = frozenset({'L1', 'A'})
    remote_go: frozenset = frozenset({'R1', 'A'})
    remote_stop: frozenset = frozenset({'R1', 'B'})
    remote_hold_s: float = 0.5
    prefetch_radius_m: float = 2.0          # request the next waypoint's fields this far before the switch
    hold_goal_tolerance_m: float = 0.5      # 'balance' starts the policy only on hold fields solved this close to here
    limits: Limits = field(default_factory=Limits)


class Fault(Exception):
    pass


class Executor:
    def __init__(self, contract, kinematics, infer: Callable, planner, fields_source, config=None, clock=time.monotonic):
        self.c, self.kin, self.infer, self.planner, self.fields_source = contract, kinematics, infer, planner, fields_source
        self.cfg = config or Config()
        self.clock = clock
        self.chain = PelvisChain(kinematics, self.cfg.mount)
        self.builder = ObservationBuilder(contract, kinematics)
        self.damp_kd = np.asarray(self.cfg.damp_kd if self.cfg.damp_kd is not None else 2 * contract['kd'], float)
        self.mode = IDLE
        self.reason = 'start'
        self.state = None; self.pose = None; self.prev_pose = None
        self.fields = None; self.fields_time = None; self.field_goal = None; self.field_hold = False; self.next_field_request = 0.
        self.inflight = None                # what the pending field request is for: ('current'|'next'|'hold', waypoint)
        self.prefetched = None              # (fields, goal, waypoint index) solved ahead of the waypoint switch
        self.mode_since = clock(); self.policy_started = None; self.builder_ready = False
        self.stand_from = None
        self.budget_violations = 0
        self.held = {}                      # remote combo -> first time seen
        self.last = {}                      # diagnostics of the last step
        self.log = []                       # per-step records (caller decides what to keep)
        self.imu_heading_hist = []          # (time, unwrapped IMU heading) for the last ~1 s
        self.heading_offset = None          # SLAM chain heading - IMU heading, at the SLAM scan time

    # ------------------------------------------------------------------ inputs
    def on_lowstate(self, state):
        if state.received is None:
            state.received = self.clock()
        self.state = state
        h = heading(quat_matrix(state.quat_wxyz))
        if self.imu_heading_hist:
            last = self.imu_heading_hist[-1][1]; h = last + (h - last + np.pi) % (2 * np.pi) - np.pi     # unwrap
        self.imu_heading_hist.append((state.received, h))
        while len(self.imu_heading_hist) > 2 and self.imu_heading_hist[0][0] < state.received - 1.0:
            self.imu_heading_hist.pop(0)

    def _imu_heading_at(self, t):
        hist = self.imu_heading_hist
        if not hist:
            return None
        times = np.array([x[0] for x in hist]); values = np.array([x[1] for x in hist])
        return float(np.interp(t, times, values))

    def on_slam_pose(self, t_map_body, stamp, received=None, age_s=None):
        """T_map_body (4x4) from FAST-LIO + localisation. stamp: scan time (any clock); received: arrival on this
        executor's clock; age_s: how long before `received` the scan was taken (clock-independent)."""
        self.prev_pose = self.pose
        received = self.clock() if received is None else received
        taken = received - (age_s or 0.)
        self.pose = dict(t=np.asarray(t_map_body, float), stamp=stamp, received=received, taken=taken)
        if self.cfg.imu_heading and self.state is not None:
            imu_then = self._imu_heading_at(taken)
            chain = self.chain.pelvis_from_body(self.pose['t'], self.state.q)[:3, :3]
            if imu_then is not None:
                self.pose['heading_offset'] = heading(chain) - imu_then

    def command(self, name):
        """Operator command: shadow | stand | balance | go | stop | idle | damp | reset. True if accepted."""
        try:
            return self._command(name)
        except Fault as exc:
            self.reason = 'refused %s: %s' % (name, exc)
            return False

    def _command(self, name):
        now = self.clock()
        if name == 'damp':
            return self._enter(DAMP, 'operator')
        if name == 'reset':
            if self.mode == DAMP:
                return self._enter(IDLE, 'operator reset after damp')
            return False
        if name == 'idle':
            if self.mode in (SHADOW, IDLE):
                return self._enter(IDLE, 'operator')
            return False                    # an active robot is never released silently: use damp
        if name == 'shadow' and self.mode == IDLE:
            self.builder_ready = False; self.planner.hold_xy = None
            return self._enter(SHADOW, 'operator')
        if name == 'stand' and self.mode == IDLE:
            if not self.cfg.allow_active:
                self.reason = 'refused: executor started without --active'; return False
            self._require_state(now)
            self.stand_from = self.state.q.copy()
            if self.last.get('pelvis_xy') is not None:     # solve "stay here" while standing: ready for 'balance'
                self.planner.hold_xy = np.asarray(self.last['pelvis_xy']); self._request_fields(now, force=True)
            return self._enter(STAND_UP, 'operator')
        if name == 'balance' and self.mode == STAND:
            # the policy takes over and stands in place (goal = here); 'go' then walks the scene
            self._check_policy_preconditions(now)
            if self.last.get('pelvis_xy') is None:
                raise Fault('no pelvis pose')
            self.builder_ready = False; self.planner.hold_xy = np.asarray(self.last['pelvis_xy'])
            self._request_fields(now, force=True)
            return self._enter(POLICY, 'operator balance')
        if name == 'go':
            if self.mode == STAND:
                self._check_policy_preconditions(now)
                self.builder_ready = False; self.planner.hold_xy = None
                return self._enter(POLICY, 'operator')
            if self.mode == POLICY and self.planner.hold_xy is not None:
                self.planner.hold_xy = None; self._request_fields(now, force=True)
                self._resume()
                return True
        if name == 'stop' and self.mode in (POLICY, SHADOW) and self.last.get('pelvis_xy') is not None:
            self.planner.hold_xy = np.asarray(self.last['pelvis_xy']); self._request_fields(now, force=True)
            return True
        self.reason = 'refused: %s in %s' % (name, self.mode)
        return False

    # ------------------------------------------------------------------ helpers
    def _enter(self, mode, reason):
        if self.mode == DAMP and mode not in (DAMP, IDLE):
            return False                    # damping is latched
        if mode != self.mode:
            self.mode, self.reason, self.mode_since = mode, reason, self.clock()
            if mode == POLICY:
                self.policy_started = self.mode_since
        return True

    def _require_state(self, now):
        s = self.state
        if s is None or now - s.received > self.cfg.limits.lowstate_timeout_s:
            raise Fault('no fresh LowState')
        if not (np.isfinite(s.q).all() and np.isfinite(s.dq).all() and np.isfinite(s.quat_wxyz).all()):
            raise Fault('non-finite LowState')
        return s

    def _check_policy_preconditions(self, now):
        if self.cfg.geofence is None:
            raise Fault('POLICY needs a geofence of real free floor (config geofence)')
        if self.fields is None:
            raise Fault('no fields yet')
        if self.pose is None or now - self.pose['received'] > self.cfg.limits.pose_timeout_s:
            raise Fault('no fresh SLAM pose')

    def _request_fields(self, now, force=False):
        """Periodic / forced refresh of the active goal's fields; prefetch of the next waypoint when close."""
        xy = self.last.get('pelvis_xy')
        if xy is None:
            return
        p = self.planner
        if p.hold_xy is not None:
            if self.mode in (STAND_UP, STAND):
                p.hold_xy = np.asarray(xy, float).copy()     # on the gantry the hold point follows the robot
            kind, index, goal = 'hold', p.active, p.hold_xy
        else:
            kind, index, goal = 'current', p.active, p.waypoints[p.active]
            close = np.linalg.norm(np.asarray(p.waypoints[p.active]) - xy) < self.cfg.prefetch_radius_m
            nxt = p.active + 1
            if not force and close and nxt < len(p.waypoints) and (self.prefetched is None or self.prefetched[2] != nxt) \
                    and self.fields is not None:
                kind, index, goal = 'next', nxt, p.waypoints[nxt]
        if force or kind == 'next' or now >= self.next_field_request:
            if self.fields_source.request(xy, goal):
                self.inflight = (kind, index)
                if kind != 'next':
                    self.next_field_request = now + p.period

    def _take_fields(self, now, fields, goal, hold=False):
        previous_goal = self.field_goal
        self.fields, self.field_goal, self.field_hold = fields, goal, hold; self.fields_time = now
        moved = previous_goal is None or np.linalg.norm(np.asarray(goal)[:2] - np.asarray(previous_goal)[:2]) > .05
        # training's stop latch keeps a stopped robot standing; a new goal (fields that arrived after the robot had
        # started stopping on the old goal's fields, as with slow field solves) resumes it, unless the scene is done
        if moved and self.builder_ready and not self.planner.reached and self.planner.hold_xy is None:
            self._resume()

    def _poll_fields(self, now):
        new = self.fields_source.poll()
        if new is None:
            return
        fields, goal = new
        kind, index = self.inflight or ('current', self.planner.active); self.inflight = None
        p = self.planner
        if kind == 'next':
            if index == p.active + 1:
                self.prefetched = (fields, goal, index)       # held until the switch
            elif index == p.active:
                self._take_fields(now, fields, goal)          # the switch happened while it was being solved
        elif kind == 'hold' and p.hold_xy is not None:
            self._take_fields(now, fields, goal, hold=True)
        elif kind == 'current' and index == p.active and p.hold_xy is None:
            self._take_fields(now, fields, goal)
        elif self.fields is None:
            self._take_fields(now, fields, goal, hold=kind == 'hold')
        # anything else is a stale answer to an outdated question: dropped

    def _advance(self, now):
        """Waypoint switched: use the prefetched fields at once if they are for this waypoint, else solve now."""
        if self.prefetched is not None and self.prefetched[2] == self.planner.active:
            fields, goal, _ = self.prefetched; self.prefetched = None
            self._take_fields(now, fields, goal)
            self.next_field_request = now + self.planner.period
        else:
            self._request_fields(now, force=True)
            if self.builder_ready:
                self._resume()

    def _resume(self):
        """Undo the training stop latch for a new goal. stop > 50: walking; 1..50: stopping (the gait phase still
        runs, so walking on keeps it); 0: standing (restart the gait in anti-phase)."""
        if self.builder.stop <= 0:
            self.builder.rearm()
        elif self.builder.stop <= 50:
            self.builder.stop = 100

    def _pelvis(self, s):
        if self.pose is None:
            return None
        imu = s.quat_wxyz if self.cfg.use_imu_tilt else None
        pos, quat, rot = self.chain.pelvis_pose(self.pose['t'], s.q, imu)
        offset = self.pose.get('heading_offset')
        if self.cfg.imu_heading and imu is not None and offset is not None and self.imu_heading_hist:
            rot = yaw_matrix(self.imu_heading_hist[-1][1] + offset - heading(rot)) @ rot
            from .frames import matrix_quat
            quat = matrix_quat(rot)
        return pos, quat, rot

    def _watchdogs(self, now, s, pelvis):
        L = self.cfg.limits
        if now - s.received > L.lowstate_timeout_s:
            raise Fault('LowState stale %.0f ms' % (1e3 * (now - s.received)))
        if not (np.isfinite(s.q).all() and np.isfinite(s.dq).all() and np.isfinite(s.gyro).all()):
            raise Fault('non-finite LowState')
        tilt = np.degrees(np.arccos(np.clip(np.abs(_gravity_z(s.quat_wxyz)), -1, 1)))
        if tilt > L.max_tilt_deg and self.mode in (STAND, POLICY):     # a robot hanging in the gantry may swing
            raise Fault('pelvis tilt %.0f deg' % tilt)
        lo, hi = self.c['soft_lower'] - L.joint_margin_rad, self.c['soft_upper'] + L.joint_margin_rad
        bad = np.flatnonzero((s.q < lo) | (s.q > hi))
        if len(bad) and self.mode in (STAND, POLICY):
            raise Fault('joint %s outside limits' % self.c['joint_order'][bad[0]])
        if s.temperature is not None and np.max(s.temperature) > L.max_motor_temp_c:
            raise Fault('motor temperature %d C' % np.max(s.temperature))
        if self.mode == POLICY:
            if self.pose is None or now - self.pose['received'] > L.pose_timeout_s:
                raise Fault('SLAM pose stale')
            if self.prev_pose is not None and self.pose is not self.last.get('checked_pose'):
                d = self.pose['t'][:3, 3] - self.prev_pose['t'][:3, 3]
                r = self.prev_pose['t'][:3, :3].T @ self.pose['t'][:3, :3]
                angle = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
                dt = max(1e-3, self.pose['taken'] - self.prev_pose['taken'])
                if np.linalg.norm(d) > L.pose_jump_m + L.pose_speed_mps * dt or angle > L.pose_jump_deg + L.pose_rate_dps * dt:
                    raise Fault('SLAM pose jump %.2f m / %.0f deg' % (np.linalg.norm(d), angle))
                self.last['checked_pose'] = self.pose
            if self.fields_time is None or now - self.fields_time > L.fields_timeout_s:
                raise Fault('fields stale')
            if pelvis is not None:
                x, y, z = pelvis[0]
                xmin, xmax, ymin, ymax = self.cfg.geofence
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    raise Fault('outside geofence (%.2f, %.2f)' % (x, y))
                if z < L.min_pelvis_z_m:
                    raise Fault('pelvis height %.2f m' % z)
            if now - self.policy_started > L.max_policy_s:
                raise Fault('policy time limit')

    def _remote(self, now, s):
        pressed = remote_buttons(s.remote)
        if self.cfg.remote_estop <= pressed:
            return 'damp'
        for name, combo in (('stand', self.cfg.remote_stand), ('go', self.cfg.remote_go), ('stop', self.cfg.remote_stop)):
            if combo <= pressed:
                first = self.held.setdefault(name, now)
                if now - first >= self.cfg.remote_hold_s:
                    self.held[name] = float('inf')        # fire once per press
                    return name
            else:
                self.held.pop(name, None)
        return None

    # ------------------------------------------------------------------ the step
    def step(self, now=None):
        now = self.clock() if now is None else now
        t0 = time.perf_counter()
        s = self.state
        cmd = None
        try:
            if s is None:
                if self.mode in ACTIVE and self.mode != DAMP:
                    raise Fault('no LowState')
                return None
            self._poll_fields(now)
            remote = self._remote(now, s)
            if remote is not None:
                self.command(remote)
            pelvis = self._pelvis(s)
            if pelvis is not None:
                self.last['pelvis_xy'] = pelvis[0][:2].copy()
                if self.planner.update(pelvis[0]):
                    self._advance(now)
                self._request_fields(now)
            if self.mode == DAMP:
                return self._damp(s)
            if self.mode in (STAND_UP, STAND, POLICY):
                self._watchdogs(now, s, pelvis)
            if self.mode == STAND_UP:
                cmd = self._stand(s, min(1., (now - self.mode_since) / self.cfg.stand_up_s))
                if now - self.mode_since >= self.cfg.stand_up_s:
                    self._enter(STAND, 'stand-up done')
            elif self.mode == STAND:
                cmd = self._stand(s, 1.)
            elif self.mode in (POLICY, SHADOW):
                cmd = self._policy(s, pelvis, now)
                if self.mode == SHADOW:
                    cmd = None                          # computed, logged, never sent
            return cmd
        except Fault as exc:
            if self.mode in ACTIVE:
                self._enter(DAMP, str(exc))
                return self._damp(s) if s is not None else None
            if self.mode == SHADOW:
                self.reason = 'shadow paused: %s' % exc; self.builder_ready = False
            return None
        finally:
            elapsed = time.perf_counter() - t0
            self.last.update(mode=self.mode, reason=self.reason, step_s=elapsed,
                             latency_s=(now - s.received + elapsed) if s is not None else None)
            if self.mode in (POLICY, STAND, STAND_UP):
                L = self.cfg.limits
                self.budget_violations = self.budget_violations + 1 if elapsed > L.step_budget_s else 0
                if elapsed > L.step_hard_limit_s or self.budget_violations >= L.step_budget_violations:
                    self._enter(DAMP, 'control step overrun %.1f ms' % (1e3 * elapsed))

    def _damp(self, s):
        return LowCmd.body(s.mode_machine, s.q, np.zeros(NUM_BODY), self.damp_kd, tau=np.zeros(NUM_BODY))

    def _stand(self, s, alpha):
        target = (1 - alpha) * self.stand_from + alpha * self.c['nominal'] if self.stand_from is not None else self.c['nominal']
        kp = self.c['kp'] * self.cfg.kp_scale * (self.cfg.stand_up_kp_start + (1 - self.cfg.stand_up_kp_start) * alpha)
        ff = np.zeros(NUM_BODY)
        if self.c['upper_gravity_compensation']:
            ff[12:] = self.kin.upper_gravity(s.q, s.quat_wxyz)
        return LowCmd.body(s.mode_machine, target, kp, self.c['kd'], tau=ff)

    def _policy(self, s, pelvis, now):
        if pelvis is None:
            raise Fault('no pelvis pose')
        if self.fields is None:
            raise Fault('no fields')
        pos, quat, rot = pelvis
        self.last['policy_wait'] = None
        if not self.builder_ready:
            hold = self.planner.hold_xy
            if hold is not None and not (self.field_hold and
                                         np.linalg.norm(np.asarray(self.field_goal)[:2] - hold) <= self.cfg.hold_goal_tolerance_m):
                self.last['policy_wait'] = 'fields of the hold point'
                return self._stand(s, 1.)               # never start on another goal's fields: the policy would walk
            obs = self.builder.reset(s.q, pos, quat, self.fields, dq=s.dq, gyro=s.gyro, pelvis_rotation=rot)
            self.builder_ready = True
        else:
            obs = self.builder.observe(s.q, s.dq, s.gyro, pos, quat, self.fields, pelvis_rotation=rot)
        action = np.asarray(self.infer(obs), float)
        if action.shape != (NUM_BODY,) or not np.isfinite(action).all():
            raise Fault('invalid policy output')
        targets = self.builder.act(action)
        ff = np.zeros(NUM_BODY)
        if self.c['upper_gravity_compensation']:
            ff[12:] = self.kin.upper_gravity(s.q, quat)
        self.last.update(action=action, targets=targets, command=self.builder.command.copy(), heading=heading(rot),
                         goal=np.asarray(self.planner.goal, float).copy())
        return LowCmd.body(s.mode_machine, targets, self.c['kp'] * self.cfg.kp_scale, self.c['kd'], tau=ff)


def _gravity_z(quat_wxyz):
    w, x, y, z = np.asarray(quat_wxyz, float) / np.linalg.norm(quat_wxyz)
    return 1 - 2 * (x * x + y * y)           # R[2, 2]: cos(tilt)
