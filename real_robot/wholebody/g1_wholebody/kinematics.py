"""Same forward kinematics as wholebody_core.WholeBodyKinematics.frames(), fast enough for the 50 Hz loop.

The original re-reads every static transform from the MJCF on each call (~6 ms on the laptop); it is called
three times per control step (pelvis chain, field sites, gravity feedforward). Here the static transforms are
taken once and the result for the last joint vector is reused. Output is identical (tests compare both).
"""
import numpy as np

from wholebody_core import WholeBodyKinematics, quat_matrix
from state_core import JOINT_NAMES


class FastKinematics(WholeBodyKinematics):
    def __init__(self, contract, path=None):
        super().__init__(contract, path)
        index = {name: i for i, name in enumerate(JOINT_NAMES)}
        self._plan = []                       # (name, parent slot, static 4x4 or None, [(joint idx, name, anchor, axis)], [(site, 4x4)])
        def visit(body, parent, is_root=False):
            joints = []
            for j in body.findall('joint'):
                a = dict(self.defaults[j.get('class')], **j.attrib)
                axis = np.fromstring(a.get('axis', '0 0 1'), sep=' '); axis = axis / np.linalg.norm(axis)
                anchor = np.fromstring(a.get('pos', '0 0 0'), sep=' ')
                joints.append((index[j.get('name')], j.get('name'), anchor, axis))
            sites = [(s.get('name'), self.transform(s)) for s in body.findall('site')]
            slot = len(self._plan)
            self._plan.append((body.get('name'), parent, None if is_root else self.transform(body), joints, sites))
            for child in body.findall('body'):
                visit(child, slot)
        visit(self.root, None, True)
        self._last_q = None; self._last = None; self._rodrigues = None
        # vectorised gravity feedforward: membership of every body in the subtree of each waist/arm joint
        self._body_names = list(self.bodies)
        self._mass = np.array([self.bodies[n]['mass'] for n in self._body_names])
        self._ipos = np.array([self.bodies[n]['ipos'] for n in self._body_names])
        self._ff_joints = JOINT_NAMES[12:]
        self._ff_body = [self.joint_body[j] for j in self._ff_joints]
        member = np.zeros((len(self._ff_joints), len(self._body_names)))
        for k, body in enumerate(self._ff_body):
            for d in self.descendants[body]:
                member[k, self._body_names.index(d)] = 1.
        self._member_mass = member * self._mass[None]
        self._total_mass = self._member_mass.sum(1)

    def frames(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (29,) or not np.isfinite(q).all():
            raise ValueError("invalid model joint vector")
        if self._last_q is not None and np.array_equal(q, self._last_q):
            return self._last
        if self._rodrigues is None:
            self._prepare_rodrigues()
        sin, cos = np.sin(q), np.cos(q)
        world = [None] * len(self._plan)
        bodies, sites = {}, {}
        eye3 = np.eye(3)
        for slot, (name, parent, static, body_joints, body_sites) in enumerate(self._plan):
            t = np.eye(4) if parent is None else world[parent] @ static
            for i, k, k2, anchor in self._rodrigues[slot]:
                r = eye3 + sin[i] * k + (1. - cos[i]) * k2                     # rotation about the joint axis
                motion = np.empty((4, 4)); motion[:3, :3] = r; motion[:3, 3] = anchor - r @ anchor; motion[3] = (0., 0., 0., 1.)
                t = t @ motion
            world[slot] = t; bodies[name] = t
            for sname, st in body_sites:
                sites[sname] = t @ st
        self._last_q = q.copy(); self._last = (bodies, sites, self._joints)
        return self._last

    def _prepare_rodrigues(self):
        self._rodrigues = []; self._joints = {}
        for name, parent, static, body_joints, body_sites in self._plan:
            entries = []
            for i, jname, anchor, axis in body_joints:
                k = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
                entries.append((i, k, k @ k, anchor)); self._joints[jname] = (anchor, axis)
            self._rodrigues.append(entries)

    def upper_gravity(self, q, base_quat):
        """Vectorised wholebody_core.WholeBodyKinematics.upper_gravity (same result, tests compare)."""
        bodies, _, joints = self.frames(q)
        rotation = quat_matrix(base_quat)
        t = np.stack([bodies[n] for n in self._body_names])                                  # [B, 4, 4]
        com = (np.einsum('bij,bj->bi', t[:, :3, :3], self._ipos) + t[:, :3, 3]) @ rotation.T     # world-rotated COMs
        jt = np.stack([bodies[b] for b in self._ff_body])                                     # [17, 4, 4]
        anchors = np.stack([joints[j][0] for j in self._ff_joints]); axes = np.stack([joints[j][1] for j in self._ff_joints])
        anchor = (np.einsum('kij,kj->ki', jt[:, :3, :3], anchors) + jt[:, :3, 3]) @ rotation.T
        axis = np.einsum('kij,kj->ki', jt[:, :3, :3], axes) @ rotation.T
        moment = self._member_mass @ com - self._total_mass[:, None] * anchor
        return -np.einsum('ki,ki->k', np.cross(moment, self.gravity), axis)
