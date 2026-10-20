"""Offline developer utility: whole-body policy contract + reference fixture from the TRAINING repository.

Run with the training environment (Click-and-Traverse-Mjlab, branch feature/mjlab-migration):
  <training>/.venv-mjlab/bin/python cat/tests/generate_wholebody_fixture.py --training-repo <training> \
      --checkpoint <training>/outputs/saved_checkpoints/generalist_finetune_v1_u596.pt

Writes (all small, committed):
  cat/wholebody/policy_contract.json   exact constants the runtime needs (joint order, nominal, soft limits, PD gains,
                                       torque limits, action rule, gait clock, hand radii, field sites, feature names)
  cat/wholebody/robot.xml              training MJCF (mesh files are NOT needed for kinematics)
  cat/tests/wholebody_reference.npz    deterministic CPU-simulator rollouts (obs, action and the raw robot state they
                                       came from) and the scene field grids, cropped to the visited region, decoded
                                       exactly as training decodes them
The runtime (cat/wholebody_core.py) must reproduce every recorded observation from the recorded raw state.
"""
import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training-repo', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--bank', default='data/furniture/demo_v1')
    p.add_argument('--scenes', nargs='+', default=['published-hurdle0', 'sidegap2-0162-plain-w0.47'])
    p.add_argument('--steps', type=int, default=160)
    args = p.parse_args()
    root = args.training_repo.resolve()
    sys.path[:0] = [str(root), str(root/'scripts')]
    import torch
    from record_mjlab_rollout import load_policy
    from cat_mjlab.runner import create_task
    from cat_mjlab.sim_cpu import CPUSimulation
    from cat_mjlab.model import assemble_training_xml
    from cat_mjlab.observation_contract import CONTRACT
    from cat_mjlab.learning import gaussian_parameters
    from cat_mjlab.packing.field_packing import torch_decode_directions
    from cat_ppo.furniture.control import JOINT_NAMES
    torch.set_num_threads(4)

    learner, saved = load_policy(str(args.checkpoint), device='cpu', policy_id=0)
    contract = saved['contract']
    env = copy.deepcopy(contract['environment_config'])
    env['push_config']['enable'] = False
    env['noise_config']['level'] = 0.
    env['dm_rand_config'].update(enable_pd=False, enable_rfi=False)
    for key in ('sgs', 'teleop', 'side_gap_curriculum'):
        env.pop(key, None)
    bank = root/args.bank
    ns = SimpleNamespace(bank_manifest=Path(str(bank)+'_packed/manifest.json'), body_collision_bank=Path(str(bank)+'_collision/manifest.json'),
                         body_collision_resets=Path(str(bank)+'_resets/manifest.json'), device='cpu', seed=0, num_envs=1, compile_task=False,
                         nconmax=contract['nconmax'], njmax=contract['njmax'], reactive_bank=None, reactive_row=None)
    task, sim, _ = create_task(ns, environment_config=env, sim_class=CPUSimulation)
    task.collision = None          # no early termination: the fixture covers whole walks

    # ------------------------------------------------------------ contract
    names = list(CONTRACT['actor_features'])+['sdf_rate.left_hand', 'sdf_rate.right_hand', 'sdf_rate.left_elbow', 'sdf_rate.right_elbow']
    assert len(names) == learner.config.actor_obs == 226
    elbow_block = [n for n in names if n.startswith('pf.elbows.')]
    upper_scales = task.upper_action_scales
    upper_scales = upper_scales.tolist() if hasattr(upper_scales, 'tolist') else [float(upper_scales)]*17
    policy = dict(
        name='generalist_finetune_v1_u596', checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        obs_dim=226, action_dim=29, feature_names=names,
        joint_order=list(JOINT_NAMES), nominal=task.nominal.tolist(), soft_lower=task.lower.tolist(), soft_upper=task.upper.tolist(),
        kp=task.kps.tolist(), kd=task.kds.tolist(), torque_limit=task.torque_limit.tolist(),
        action_scale=float(env.get('action_scale', .5)), upper_action_scales=upper_scales, upper_target_rate=float(env.get('upper_target_rate', 2.)),
        upper_gravity_compensation=bool(env.get('upper_gravity_compensation', False)),
        control_dt=float(task.dt), physics_dt=float(sim.model.opt.timestep), substeps=int(task.n_substeps),
        hand_radii=task.hand_radii.tolist(), elbow_sdf_offset=.05, odometry_period_steps=5,
        gait_bound=float(env.get('gait_config', {}).get('gait_bound', .6)),
        foot_height=float(np.mean(env.get('gait_config', {}).get('foot_height_range', [.07, .07]))),
        frequency_range=list(env.get('gait_config', {}).get('freq_range', [1.3, 1.5])),
        keep_yaw=bool(env.get('yaw_command') or env.get('heading_command')),
        field_sites=list(task.site_names), elbow_sites=['left_elbow_probe', 'right_elbow_probe'],
        elbow_df_index=[elbow_block.index(n) for n in elbow_block if n.startswith('pf.elbows.df.')],
        gravity=[float(v) for v in sim.model.opt.gravity],
        field_convention='grid points at origin + i*dx; trilinear with x-fast corners and z-fast weights (training quirk); '
                         'guidance/boundary are unit vectors, sdf in metres',
        source='Click-and-Traverse-Mjlab cat_mjlab/task.py (step, _fields, _elbow_fields, _observe, reset), task_math.py')
    out = HERE/'wholebody'
    out.mkdir(exist_ok=True)
    (out/'policy_contract.json').write_text(json.dumps(policy, indent=1)+'\n')
    xml = assemble_training_xml()
    xml = re.sub(r'file="[^"]*/([^"/]+)"', r'file="assets/\1"', xml)        # meshes not shipped; FK needs none
    (out/'robot.xml').write_text(xml)

    # ------------------------------------------------------------ rollouts
    index = {s['scene_id']: i for i, s in enumerate(task.bank.manifest['scenes'])}
    rec = {k: [] for k in ('scene', 'step', 'qpos', 'qvel', 'gyro', 'obs', 'action', 'sites', 'elbow_sites', 'pelvis_rotation', 'sites_final', 'done')}
    grids = {}
    for scene_number, scene in enumerate(args.scenes):
        task.reset(torch.zeros(1, dtype=torch.long), torch.tensor([index[scene]]))
        points = []
        for t in range(args.steps):
            obs = task.obs['state']
            with torch.no_grad():
                mean, _ = gaussian_parameters(learner.model.logits(obs, torch.zeros(1, dtype=torch.long)))
            action = torch.tanh(mean)
            rec['scene'].append(scene_number); rec['step'].append(t)
            rec['qpos'].append(sim.data.qpos[0].numpy().copy()); rec['qvel'].append(sim.data.qvel[0].numpy().copy())
            rec['gyro'].append(task._sensor('gyro_pelvis', task.all_ids)[0].numpy().copy())
            rec['obs'].append(obs[0].numpy().copy()); rec['action'].append(action[0].numpy().copy())
            rec['sites'].append(sim.data.site_xpos[0, task.site_ids].numpy().copy())
            rec['elbow_sites'].append(sim.data.site_xpos[0, task.elbow_ids].numpy().copy())
            # MuJoCo computes site poses BEFORE the last integration substep: training sampled fields there (2 ms
            # behind qpos). Keep those (sites, pelvis_rotation) for the exact replay, and the kinematics AT qpos
            # (sites_final) for the forward-kinematics test.
            rec['pelvis_rotation'].append(sim.data.site_xmat[0, task.pelvis_site].numpy().reshape(3, 3).copy())
            import mujoco
            sim._shadow.qpos[:] = sim.data.qpos[0].double().numpy(); mujoco.mj_kinematics(sim.model, sim._shadow)
            rec['sites_final'].append(np.vstack((sim._shadow.site_xpos[task.site_ids.numpy()], sim._shadow.site_xpos[task.elbow_ids.numpy()])).copy())
            points.append(rec['sites'][-1]); points.append(rec['elbow_sites'][-1])
            points.append(task.info['odom_delay'][0, :3].numpy()[None])
            rec['done'].append(bool(task.step(action)['done'][0]))
        # Field grid cropped to the visited region (+ margin covering held-odometry offsets)
        s = index[scene]
        X, Y, Z = (int(v) for v in task.bank.shapes[s]); dx = float(task.bank.dxs[s]); origin = task.bank.origins[s].numpy().astype(np.float64)
        offset = int(task.bank.offsets[s])
        pts = np.concatenate(points)
        lo = np.maximum(np.floor((pts.min(0)-origin)/dx).astype(int)-12, 0); hi = np.minimum(np.ceil((pts.max(0)-origin)/dx).astype(int)+12, [X, Y, Z])
        lo[2], hi[2] = 0, Z
        full = lambda name: task.bank.fields[name][offset:offset+X*Y*Z]
        def crop(values, width):
            v = values.reshape(X, Y, Z, *width)
            return v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        decode = (lambda name: torch_decode_directions(full(name)).numpy()) if 'gf' in getattr(task.bank, 'packed_fields', ()) else (lambda name: full(name).numpy())
        grids[scene_number] = dict(sdf=crop(full('sdf').numpy(), ()).astype(np.float32), gf=crop(decode('gf'), (3,)).astype(np.float32),
                                   bf=crop(decode('bf'), (3,)).astype(np.float32), origin=origin+lo*dx, dx=dx, name=scene)
    payload = {k: np.asarray(v) for k, v in rec.items()}
    for n, g in grids.items():
        for k in ('sdf', 'gf', 'bf', 'origin'):
            payload[f'grid{n}_{k}'] = g[k]
        payload[f'grid{n}_dx'] = np.array(g['dx'])
    payload['scene_names'] = np.array([grids[n]['name'] for n in sorted(grids)])
    np.savez_compressed(HERE/'tests'/'wholebody_reference.npz', **payload)
    print(json.dumps(dict(steps=len(rec['obs']), scenes=args.scenes, grids={n: list(g['sdf'].shape) for n, g in grids.items()},
                          bytes=(HERE/'tests'/'wholebody_reference.npz').stat().st_size)))


if __name__ == '__main__':
    main()
