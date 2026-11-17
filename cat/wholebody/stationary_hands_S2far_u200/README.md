# stationary_hands_S2far_u200 — standing hand protection

When an object approaches a hand while the robot stands, it pulls that hand away and keeps its feet planted;
otherwise it walks like the generalist. Lineage: `robust_delay_v1_u150` → S2 (approaching objects at the standing
robot, joint mechanical power priced at -0.02 per W in standing episodes, no stance rule) → S2far (hand clearance
target 0.20 m, anticipation 0.40 m, at least 0.15 m retreat for threatening objects), 200 updates.

## What it does (simulation)

| Standing, 2000 approaching-object events | before (generalist) | this policy |
|---|---|---|
| mean hand retreat from a threatening object | 12.9 cm | 25.2 cm |
| threats met with a retreat of 15 cm or more | 37% | 90% |
| danger / anticipation events handled | 0.87 / 0.72 | 0.99 / 0.98 |
| episodes where a foot moved more than 5 cm | 49% | 3% |
| joint mechanical power while standing | 43.7 W | 30.8 W |

Traversal kept: 0.60 clean goal-reaching on 384 non-table scenes (hurdles, beams, side gaps, cabinet passages,
clutter, furniture).

## Limits

* **The robot's fields do not see moving objects.** On the robot the obstacle fields are solved per waypoint from the
  map (about 2.5 s on the Jetson), so an object swung at a hand never enters them: the standing reflex this policy
  learned cannot be shown with the current perception. With static scenes it behaves as a walking policy.
* Flinches at objects that would only pass by (leaves them alone in 10% of events; the generalist 59%).
* Turns sideways at narrow table pairs (83% of near-table steps); use `stationary_hands_T2_u410` for tables.

## Run

Same runtime, contract and procedure as `robust_delay_v1_u150` (only `policy.onnx` differs): `--policy stationary_hands_S2far_u200` on
the executor, e.g. `g1.sh wb shadow --policy stationary_hands_S2far_u200 --scene <scene> --anchor-at-robot` first (motors untouched), then
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
from the training repository's deploy bundle `outputs/deploy/hands_S2far_u200/` (Click-and-Traverse-Mjlab, branch
feature/mjlab-migration; full bundle with deploy_spec, rollouts and TorchScript); checkpoint
`outputs/saved_checkpoints/stationary_hands_S2far_u200.pt`, sha256 `d29faa3f4c42553751ad64b92614e42ec8f1341ccee489af0448cfc32edad9f5`. Training log: `docs/stationary_hands_20261113.md`.
