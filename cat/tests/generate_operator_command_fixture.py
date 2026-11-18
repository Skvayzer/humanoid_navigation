"""Offline developer utility: reference commands for the operator-speed rule from the TRAINING code.

Run with the training environment (Click-and-Traverse-Mjlab, branch feature/mjlab-migration, commit 62ac46a or later):
  <training>/.venv-mjlab/bin/python cat/tests/generate_operator_command_fixture.py --training-repo <training>
Writes cat/tests/operator_command_reference.npz: random field samples (unit guidance / boundary at the 11 field sites,
including near-cancelling projections), operator speeds, and the commands of task.CATTask._operator_command (speed
rule) and task_math.compute_cmd_from_rtf (original rule) on them. test_wholebody checks wholebody_core against both.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-repo', type=Path, required=True)
    p.add_argument('--cases', type=int, default=400)
    args = p.parse_args()
    sys.path.insert(0, str(args.training_repo.resolve()))
    import torch
    from cat_mjlab.task import CATTask
    from cat_mjlab import task_math
    g = torch.Generator().manual_seed(0); n = args.cases
    unit = lambda x: x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-9)
    gf = unit(torch.randn((n, 11, 3), generator=g, dtype=torch.float64))
    bf = unit(torch.randn((n, 11, 3), generator=g, dtype=torch.float64))
    forward = torch.rand(n, generator=g) < .5          # half the cases: root guidance mostly forward, as in a passage
    gf[forward, 1] = unit(torch.tensor([1., 0., 0.], dtype=torch.float64) + .3 * torch.randn((int(forward.sum()), 3), generator=g, dtype=torch.float64))
    speed = .1 + .5 * torch.rand(n, generator=g, dtype=torch.float64)
    fake = SimpleNamespace(op_speed=speed, operator_speed=dict(cutout=.05))
    operator = CATTask._operator_command(fake, gf, bf, torch.arange(n))
    legacy = task_math.compute_cmd_from_rtf(gf[:, 1], gf[:, [0, 3, 4, 5, 6]], bf[:, [0, 3, 4, 5, 6]])
    out = HERE / 'operator_command_reference.npz'
    np.savez_compressed(out, gf=gf.numpy(), bf=bf.numpy(), speed=speed.numpy(), operator=operator.numpy(), legacy=legacy.numpy())
    stops = int((operator[:, 0] < .5).sum())
    print(f'{out}: {n} cases, {stops} operator stops, {int((legacy[:, 0] < .5).sum())} original-rule stops')


if __name__ == '__main__':
    main()
