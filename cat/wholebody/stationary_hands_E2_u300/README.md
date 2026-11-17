# stationary_hands_E2_u300 — all-round, energy-priced

The best traversal of the hand-protective policies and the lowest energy per metre: joint mechanical power is priced
everywhere (-0.02 per W at tables, -0.01 per W elsewhere) instead of hand-written posture rules. Lineage:
`robust_delay_v1_u150` → S2 → S2far u200 → E2 (energy price, table-gap curriculum), 300 updates.

## What it does (simulation)

| 384 non-table scenes, clean goal-reaching | this policy |
|---|---|
| all | 0.62 |
| clutter / furniture rooms | 0.91 / 0.89 |
| CAT obstacle scenes (hurdles, beams, overhead) | 0.37 |
| side gaps | 0.30 |
| narrow cabinet passages < 0.60 m | 0.11 |

Table pairs: goal reached 53% (hands raised in 45% of near-table steps, turned sideways in 41%). Mechanical cost of
transport 0.875. Standing hand protection is inherited from S2far and was not re-measured.

## Limits

* The main failure in simulation is touching an obstacle (26% of episodes), not stopping (13%).
* For tables `stationary_hands_T2_u410` is stronger; for standing protection see `stationary_hands_S2far_u200`.

## Run

Same runtime, contract and procedure as `robust_delay_v1_u150` (only `policy.onnx` differs): `--policy stationary_hands_E2_u300` on
the executor, e.g. `g1.sh wb shadow --policy stationary_hands_E2_u300 --scene <scene> --anchor-at-robot` first (motors untouched), then
`active` on the gantry with the usual checklist. The robot image copies `cat/wholebody/` whole, so rebuild or reload
the wholebody image (`g1.sh build wholebody`, or `laptop.sh build-robot` + `g1.sh load`) to get this folder on the robot.

## Robustness and status

Trained with the same randomisation as `robust_delay_v1_u150` (action delay 0-20 ms, SLAM-rate pose noise, LiDAR-mount
error, friction x0.5-1.2, mass x0.9-1.1, gain/torque/observation noise, pushes; `policy_contract.json` → `robustness`),
starting from that policy. **Not yet sim-to-sim tested and not yet evaluated under the training noise**: all numbers
above are deterministic, noise-free simulation on training scenes, one seed. Start supervised.

## Provenance

`policy_contract.json` differs from `../policy_contract.json` only in `name`, `checkpoint_sha256`, `policy_onnx` and
`robustness`; `cat/tests/test_wholebody.py` checks that, the ONNX hash, and that the ONNX reproduces the training actor
on `reference_actions.npz` (320 steps of its CPU-simulator rollouts). Made with `cat/tests/ship_wholebody_variant.py`
from the training repository's deploy bundle `outputs/deploy/hands_E2_u300/` (Click-and-Traverse-Mjlab, branch
feature/mjlab-migration; full bundle with deploy_spec, rollouts and TorchScript); checkpoint
`outputs/saved_checkpoints/stationary_hands_E2_u300.pt`, sha256 `5933b120b989a33eba59660a3344a5ccfa94df30b6ef0bf480c200f53d9c893d`. Training log: `docs/stationary_hands_20261113.md`.
