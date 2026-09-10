"""Offline utility: emit reference samples from an explicitly supplied CAT pf.py."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import skfmm

source = Path(sys.argv[1]).read_bytes()
names = {'PFConfig', 'make_axes', 'make_grid', 'make_sdf', 'grad3',
         'make_guidance_field_progressive', 'make_pf_for_octomap'}
parsed = ast.parse(source)
selected = ast.Module(body=[n for n in parsed.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                           and n.name in names], type_ignores=[])
namespace = dict(np=np, skfmm=skfmm)
exec(compile(selected, 'pinned-upstream-pf', 'exec'), namespace)
cfg = namespace['PFConfig']()
cfg.origin_w = np.zeros(3)
cfg.goal_w = np.array([3.5, 2.5, .75])
occupied = np.zeros((128, 128, 35), bool)
occupied[:20] = True
sdf, bf, gf = namespace['make_pf_for_octomap'](cfg, occupied)
indices = [(x, y, z) for x in (20, 22, 30, 60, 87, 100) for y in (40, 62, 80) for z in (5, 18, 30)]
samples = [dict(index=p, sdf=float(sdf[p]), boundary=bf[p].tolist(), guidance=gf[p].tolist()) for p in indices]
print(json.dumps(dict(source_sha256=hashlib.sha256(source).hexdigest(),
                      skfmm_version=skfmm.__version__, samples=samples), indent=2))
