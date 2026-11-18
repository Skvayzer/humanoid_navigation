"""Offline developer utility: ship a whole-body policy VARIANT next to the existing ones (nothing is replaced).

Input: a deploy bundle written by the training repository (Click-and-Traverse-Mjlab, scripts/export_deploy_bundle.py):
  <bundle>/policy.onnx, deploy_spec.json, reference_rollouts.npz
Writes cat/wholebody/<name>/ (refuses if it exists):
  policy.onnx            copied unchanged
  policy_contract.json   the main contract (cat/wholebody/policy_contract.json) with this policy's name, checkpoint hash,
                         ONNX record and robustness record; every runtime constant stays identical, so
                         cat/wholebody_core.py and the robot executor run it unchanged
  reference_actions.npz  --steps observations spread over the bundle's CPU-simulator rollouts and the training actor's
                         actions on them (cat/tests/test_wholebody.py checks that the ONNX reproduces them)
The README.md next to it is written by hand.

Refuses unless (1) the bundle's 226 feature names equal the main contract's, (2) with --same-runtime-as, every runtime
section of deploy_spec.json (policy, timing, joints, action, actuation, observation, fields, gait, training
randomisation) equals that bundle's, and (3) ONNX Runtime reproduces the recorded actions within 1e-5.

  .venv-sim/bin/python cat/tests/ship_wholebody_variant.py --bundle <training>/outputs/deploy/hands_T2_u410 \
      --name stationary_hands_T2_u410 --same-runtime-as <training>/outputs/deploy/generalist_robust_delay_v1_u150 \
      --robustness-from cat/wholebody/robust_delay_v1_u150/policy_contract.json --marked '...' --parent '...' \
      --exported-from '...'
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RUNTIME_SECTIONS = ('policy', 'timing', 'joints', 'action', 'actuation', 'observation', 'fields', 'gait', 'training_randomisation')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--name', required=True, help='folder under cat/wholebody/ and the contract name')
    p.add_argument('--same-runtime-as', type=Path, help='a deploy bundle already run on the robot')
    p.add_argument('--robustness-from', type=Path, required=True, help='contract whose training_randomisation record applies')
    p.add_argument('--marked', required=True, help='one line: what this policy is for')
    p.add_argument('--parent', required=True, help='training lineage')
    p.add_argument('--exported-from', required=True, help='checkpoint path(s) in the training repository')
    p.add_argument('--steps', type=int, default=320)
    p.add_argument('--operator-speed', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='the policy was trained with an operator speed in [MIN, MAX] m/s: record it, so the executor demands --speed')
    p.add_argument('--goal-seed', choices=('ball', 'column'),
                   help='goal seed the policy was trained with (column = goal projected onto the ground); default: the one '
                        'recorded in deploy_spec.json environment_config, ball if none')
    args = p.parse_args()

    out = HERE/'wholebody'/args.name
    if out.exists():
        raise SystemExit(f'{out} exists: variants are never replaced, pick a new name')
    spec = json.loads((args.bundle/'deploy_spec.json').read_text())
    trained_seed = ((spec.get('environment_config') or {}).get('goal_seed') or {}).get('mode', 'ball')
    if args.goal_seed is not None and args.goal_seed != trained_seed:
        raise SystemExit(f'--goal-seed {args.goal_seed}: the policy was trained with the {trained_seed} goal seed')
    args.goal_seed = trained_seed
    main_contract = json.loads((HERE/'wholebody'/'policy_contract.json').read_text())
    rollouts = np.load(args.bundle/'reference_rollouts.npz')

    names = [str(n) for n in rollouts['feature_names']]
    if names != main_contract['feature_names']:
        raise SystemExit('observation feature names differ from cat/wholebody/policy_contract.json')
    if args.same_runtime_as is not None:
        reference = json.loads((args.same_runtime_as/'deploy_spec.json').read_text())
        differ = [s for s in RUNTIME_SECTIONS if spec.get(s) != reference.get(s)]
        if differ:
            raise SystemExit(f'runtime sections differ from {args.same_runtime_as}: {differ}')

    obs, action = rollouts['obs'].astype(np.float32), rollouts['action'].astype(np.float32)
    index = np.unique(np.linspace(0, len(obs)-1, min(args.steps, len(obs))).round().astype(int))
    obs, action = obs[index], action[index]
    import onnxruntime
    session = onnxruntime.InferenceSession(str(args.bundle/'policy.onnx'), providers=['CPUExecutionProvider'])
    error = float(np.abs(session.run(None, {'obs': obs})[0]-action).max())
    if error > 1e-5:
        raise SystemExit(f'ONNX does not reproduce the recorded actions: max error {error:.2e}')

    onnx = (args.bundle/'policy.onnx').read_bytes()
    robustness = json.loads(args.robustness_from.read_text())['robustness']
    contract = dict(main_contract)
    contract.update(name=args.name, checkpoint_sha256=spec['provenance']['checkpoint_sha256'])
    contract['policy_onnx'] = dict(
        file='policy.onnx', sha256=hashlib.sha256(onnx).hexdigest(), size=len(onnx),
        inputs='obs float32 [B, 226]', outputs='action = tanh(mean) float32 [B, 29], mean float32 [B, 29]',
        exported_from=args.exported_from+' (Click-and-Traverse-Mjlab scripts/export_deploy_bundle.py, '
                      f"training commit {spec['provenance']['git_commit'][:7]})")
    contract['robustness'] = dict(
        marked=args.marked, parent=args.parent, training_randomisation=robustness['training_randomisation'],
        same_as=f"{args.robustness_from.parent.name}: deploy_spec training randomisation identical",
        sim2sim_plain_mujoco='not run for this policy',
        limits=robustness['limits'])
    if args.operator_speed:
        low, high = args.operator_speed
        contract['command'] = dict(rule='operator_speed', trained_speed_m_s=[low, high], cutout_m_s=.05, nominal_m_s=.525,
                                   note='field direction at the operator speed (executor --speed), no stop latch; '
                                        'wholebody_core.command_from_fields(speed=...) = task._operator_command')
    if args.goal_seed != 'ball':
        contract['goal_seed'] = args.goal_seed
    out.mkdir(parents=True)
    shutil.copyfile(args.bundle/'policy.onnx', out/'policy.onnx')
    (out/'policy_contract.json').write_text(json.dumps(contract, indent=1)+'\n')
    np.savez_compressed(out/'reference_actions.npz', obs=obs, action=action)
    print(json.dumps(dict(out=str(out), steps=len(index), onnx_max_abs_err=error, onnx_sha256=contract['policy_onnx']['sha256'],
                          checkpoint_sha256=contract['checkpoint_sha256']), indent=1))


if __name__ == '__main__':
    main()
