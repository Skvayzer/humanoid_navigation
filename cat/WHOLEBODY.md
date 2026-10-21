# Whole-body CAT policy: robot-side runtime and sim-to-sim (no motion)

Policy `generalist_finetune_v1_u596` (Click-and-Traverse-Mjlab): one network for hurdles, crouching under beams,
sideways gaps, rooms. **226 observations -> 29 joint actions (whole body)** -- not the released 162 -> 12 legs-only
model that `checkpoint_manifest.json` pins. Nothing here commands motors.

## Policies

| Folder | Policy | Use |
|---|---|---|
| `wholebody/` | `generalist_finetune_v1_u596` | original; no action-latency training |
| `wholebody/robust_delay_v1_u150/` | `generalist_robust_delay_v1_u150` | **action-delay and perturbation resistant** (0-20 ms delay, pose/mount noise, friction, mass); same contract and runtime -- see its README |

## Files

| File | What |
|---|---|
| `wholebody/policy_contract.json` | generated from the checkpoint: joint order, nominal pose, soft limits, kp/kd, torque limits, action rule, gait clock, hand radii, 13 field sites, all 226 feature names, checkpoint SHA-256 |
| `wholebody/robot.xml` | the training MJCF (kinematics, sites, masses); meshes not needed for FK |
| `wholebody_core.py` | numpy runtime: field sampler (training interpolation), FK for all 13 sites, waist/arm gravity feedforward, action -> targets, 226-input observation builder |
| `wholebody_fields.py` | occupancy -> fields exactly as the training generator (unit directions, outward normal inside obstacles, voxel centres); unknown space must be passed as occupied |
| `wholebody_sim.py` | closed-loop sim-to-sim against plain MuJoCo (development only) |
| `tests/test_wholebody.py` + `tests/wholebody_reference.npz` | recorded training-simulator steps; the runtime must rebuild them |
| `tests/generate_wholebody_fixture.py` | regenerates contract, robot.xml and fixture from the training repository |
| `wholebody/scenes/*.json` | sim scenes (training scenes with box lists, runtime frame) |

The policy itself (`policy.onnx`) is not committed: deploy bundle `outputs/deploy/generalist_finetune_v1_u596/`
in the training repository (SHA256SUMS there; contract carries the checkpoint hash).

## Runtime interface (one control step, 50 Hz)

```python
targets = builder.act(action)            # policy output (tanh, 29) -> joint targets; send q=targets, kp, kd,
                                         # tau = feedforward (joints 12..28: kin.upper_gravity(q, imu_quat))
# ... 20 ms later, with the newest state:
obs = builder.observe(q, dq, gyro, pelvis_pos_map, pelvis_quat_map, fields)    # float32 [226]
action = onnx_session.run(None, {'obs': obs[None]})[0][0]
```
`fields` = `wholebody_fields.training_fields(blocked, first_voxel_centre, 0.04, goal_xyz)` on the current OctoMap
window (occupied OR unknown = blocked), goal at z = 0.75 m, recomputed ~1 Hz. `builder.rearm()` after a new goal
when the robot had stopped (training's stop latch keeps a stopped robot standing).

## Validation (all on dep-1)

* 320 recorded training-simulator steps (hurdle + 0.47 m side gap) rebuilt from raw state: max error **2.0e-5**
  over all 226 inputs; ONNX Runtime actions on the rebuilt inputs vs recorded: **2.0e-6**.
* FK vs MuJoCo at the same state: < 0.02 mm (all 13 sites). Gravity feedforward vs MuJoCo `qfrc_bias`: matches.
* Fields from occupancy vs the training scenes' fields: side gap identical (cos 1.0000); released hurdle and a
  procedural scene median cos 0.9996-0.9997, sdf < 0.5 mm at the visited body points.
* Note: MuJoCo reports site poses before the last integration substep, so training sampled fields ~2 ms behind
  the joint state; on the robot FK uses the current state (difference < 1 cm, fast limbs only).
* Full repository suite (with the native OctoMap module built against OctoMap 1.9.8): 83 tests OK, 13 skipped (ROS).

## Sim-to-sim (plain MuJoCo, physical obstacles, contacts counted)

| Condition | flat 4 m | gap 0.47 | corridor 0.48 | gap 0.44 (exact) |
|---|---|---|---|---|
| nominal | reached | reached, 0 contacts | reached, 0 contacts | reached, 0 contacts |
| each factor alone (pose noise 1 cm / 0.01 rad, pose latency 40 ms, action delay 20 ms, kp -10%, perception limits) | 18/18 | 18/18, 0 contacts | -- | -- |
| **realistic combined** (pose noise, 40 ms pose latency, **10 ms** action latency, kp -10%, perception limits), 5 seeds | **5/5** | **5/5** | **5/5** | **5/5** |
| combined with a **full 20 ms action delay** | 3-4/5, 2-3 falls | 2-5/5 | 2-3/5 | 4/5 |
| 25 m mixed course, nominal | -- | -- | -- | falls at 5.5 s, 0 contacts |

Findings:
1. The runtime transfers: under realistic perturbations, 20/20 through gaps down to 0.44 m with no contact.
2. **Latency is the main risk.** Training never randomised action latency (`enable_ctrl_delay: false`); a full
   control period of delay plus pose noise causes falls. Measure observation-to-motor latency on the robot (target
   well below 20 ms) or fine-tune with action-delay randomisation before hardware.
3. On the 25 m corridor course the policy walks sideways along the corridor and falls near a low beam -- the same
   happens in the training simulator, so it is the policy, not the transfer. Use goals within open space first.

Delay options of `wholebody_sim.py`: `--action-latency-ms MS` (0-40) = motor-side latency, the model used in training
and the one that matches the robot runtime; `--action-delay N` = the runtime applies an N-steps-old action (its own
recorded last action and targets go stale), a harsher failure mode.

Commands:
```bash
PYTHONPATH=cat python -m unittest cat.tests.test_wholebody -v
PYTHONPATH=cat python cat/wholebody_sim.py --bundle <bundle> --scene sidegap_plain_047 \
    --pose-noise 0.01 --pose-latency 2 --action-latency-ms 10 --kp-scale 0.9 --robot-perception --video out.mp4
```

## Still required before any motion (separate review)

1. Pelvis pose in the OctoMap/map frame at >= 10 Hz: SLAM body pose x calibrated LiDAR->torso x waist FK
   (inverse) -- not yet verified on the robot.
2. Live field node: OctoMap window -> `training_fields` (unknown = blocked) -> FieldGrid, ~1 Hz, goal from the
   preview goal topic / Nav2.
3. Shadow runner: LowState + pose + fields -> observations -> ONNX -> logged actions, no publisher.
4. LowCmd publisher (q, kp, kd, tau feedforward), hand-over from the Unitree controller (damping -> nominal stand ->
   policy), E-stop to damping, watchdogs (stale LowState / pose / fields, pose jumps), measured latency.
5. Gantry, then crane, then floor -- as in the deployment guide.
