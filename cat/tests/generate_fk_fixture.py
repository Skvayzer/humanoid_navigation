"""Offline developer utility: emit MuJoCo FK reference JSON to stdout."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from state_core import SITE_NAMES, JOINT_NAMES

root = ET.parse(str(Path(__file__).resolve().parents[1]/'upstream/g1_mjx_feetonly_torque.xml')).getroot()
# Only visual/collision assets removed; all inertias, joints, sites, body
# transforms and class inheritance remain exactly those of the upstream model.
for parent in list(root.iter()):
    for child in list(parent):
        if child.tag in ('asset', 'geom', 'contact'):
            parent.remove(child)
model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
assert [model.joint(i).name for i in range(1, model.njnt)] == list(JOINT_NAMES)
cases = []
for q in (np.zeros(29), np.linspace(-.08, .08, 29), np.linspace(.1, -.05, 29)):
    data = mujoco.MjData(model)
    data.qpos[:3] = 0
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = q
    mujoco.mj_forward(model, data)
    sites = {}
    for name in SITE_NAMES:
        sid = model.site(name).id
        matrix = np.eye(4)
        matrix[:3, :3] = data.site_xmat[sid].reshape(3, 3)
        matrix[:3, 3] = data.site_xpos[sid]
        sites[name] = matrix.tolist()
    cases.append(dict(q=q.tolist(), sites=sites))
print(json.dumps(dict(generator='MuJoCo '+mujoco.__version__, cases=cases), indent=2))
