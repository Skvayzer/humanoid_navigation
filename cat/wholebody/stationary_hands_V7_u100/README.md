# stationary_hands_V7_u100 — slow traversal with speed control

The first policy trained both to follow an operator speed and to stop at the goal. The walking direction comes from the
fields as before; the speed is the operator's `--speed` (trained 0.1-0.6 m/s; the executor refuses this policy without
it). The goal is projected onto the ground: the guidance is seeded from a vertical column at the goal, not from a ball
at 0.75 m, which had acted as a hidden distance-to-goal gauge (the old policies stopped about 0.4 m short of every
goal). The contract records `goal_seed: column`, so the executor builds its fields that way automatically. Success in
training counted only within 0.30 m of the goal after a 2 s hold, with the same arrival stop as the executor (command
zero within 0.25 m of the final goal, back on beyond 0.45 m). Lineage: `robust_delay_v1_u150` → S2 → S2far u200 →
E2 u300 → V5 u100 → V6 u100 → V7, 100 updates (training continues; later snapshots may follow).

```
python -m g1_wholebody.ros_executor --policy stationary_hands_V7_u100 --speed 0.15 ...
```

The contract also records `upper_filter: avg2`: V7 was trained with the waist/arm motors tracking the mean of the
last two targets (`--upper-target-average`), so the executor applies the same `avg2` filter unless `--upper-filter`
says otherwise (it warns when that differs from training). Do not run it with the `ema 0.3` filter used for u150.

## What it does (robot-side simulator, solid walls, told 0.15 m/s unless stated)

| Test | V7 u100 | V6 u100 (before) |
|---|---|---|
| Open-floor goals 0.5-2 m ahead: closest approach | 0.11-0.19 m (reached) | 0.29-0.40 m (short) |
| Goals 0.5-0.8 m before an opening, or 0.3 m past it | 0.06-0.23 m (reached) | 0.14-0.39 m |
| Goals 0.2-0.35 m before an opening or at its mouth | 0.25-0.30 m, then backs away to 0.31-0.40 m | 0.33-0.40 m |
| 0.47-0.48 m gaps, speed near the walls, told 0.15 m/s | 0.24-0.28 m/s (peaks 0.45-0.58) | 0.13-0.25 m/s |
| The same, told 0.3 m/s | 0.30-0.33 m/s (peaks 0.49-0.56) | |
| Wall contacts in those gap passages | 1 of 6 (left wrist, 0.3 s) | 3 of 3 at 0.15 m/s |
| Falls | none | none |

## Limits

- **Goals right next to an opening are not reached:** at 0.2-0.35 m in front of a passage or at its mouth it stops
  0.25-0.30 m away and backs off. This is learned hand clearance (its palms keep 10-18 cm from walls); put goals at
  least 0.5 m from walls. A fine-tune with goals beside walls is planned.
- At 0.15 m/s it still moves about 1.7x the command near walls; at 0.3 m/s it follows the command.
- It can brush a wall with the left wrist in a slow 0.47 m passage.
- Never put the goal inside a narrow gap: standing still inside a gap was never trained.

## Robustness and status

Trained with the same randomisation as `robust_delay_v1_u150` (action delay 0-20 ms, SLAM-rate pose noise, LiDAR-mount
error, friction x0.5-1.2, mass x0.9-1.1, gain/torque/observation noise, pushes). **Not yet run on the robot and not yet
evaluated under the training noise**: the numbers above are deterministic, noise-free simulation, one seed. Start in
`shadow`, then `active` on the gantry, at `--speed 0.15`.

## Provenance

`policy_contract.json` differs from `../policy_contract.json` only in `name`, `checkpoint_sha256`, `policy_onnx`,
`robustness`, `command` (operator speed 0.1-0.6 m/s), `goal_seed` (column) and `upper_filter` (avg2); `cat/tests/test_wholebody.py` checks
that, the ONNX hash, and that the ONNX reproduces the training actor on `reference_actions.npz` (320 steps; max error
6.6e-7). Made with `cat/tests/ship_wholebody_variant.py --same-runtime-as hands_E2_u300 --operator-speed 0.1 0.6` from
the training repository's deploy bundle `outputs/deploy/slow_V7_u100/` (Click-and-Traverse-Mjlab, branch
feature/mjlab-migration); checkpoint `outputs/saved_checkpoints/slow_V7_u100.pt` (cluster run
`stationary_hands_20261113_V7`, snapshot `update_0100`), sha256
`f57a8833ddb05bfde631553909d592fc514a83e26abfc0d6600e6a78693962ef`. Training log: `docs/stationary_hands_20261113.md`.
