# stationary_hands_S2far2_u330 — standing hand protection, wider margins

The S2far behaviour with larger safety margins: when an object approaches a hand while the robot stands, it pulls that
hand farther away (target 0.30 m, anticipation 0.50 m, at least 0.25 m retreat for threatening objects) and keeps its
feet planted. Lineage: `robust_delay_v1_u150` → S2 → S2far u200 → S2far2, 330 updates.

## What it does (simulation)

| Standing, 2000 approaching-object events | generalist | S2far u200 | S2far (final) | this policy |
|---|---|---|---|---|
| mean hand retreat from a threatening object | 12.9 cm | 22.1 cm | 25.2 cm | **26.8 cm** |
| retreat >= 15 cm / >= 25 cm | 37% / 10% | 84% / 35% | 90% / 52% | **93% / 62%** |
| danger / anticipation events handled | 0.87 / 0.72 | 0.99 / 0.96 | 0.99 / 0.98 | **0.99 / 0.99** |
| pass-by object left alone (no flinch) | 0.59 | 0.15 | 0.10 | 0.07 |
| episodes where a foot moved more than 5 cm | 49% | 5.0% | 3.3% | 5.8% |
| joint mechanical power while standing | 43.7 W | 32.1 W | 30.8 W | 31.1 W |

## Limits

* **The robot's fields do not see moving objects.** Fields are solved per waypoint from the map (about 2.5 s on the
  Jetson), so an object swung at a hand never enters them: the standing reflex cannot be shown with the current
  perception. With static scenes it behaves as a walking policy.
* Flinches at objects that would only pass by even more than S2far (leaves them alone in 7% of events).
* Traversal (tables, gaps, rooms) was not re-evaluated for this checkpoint; for tables use `stationary_hands_T2_u410`.

## Run

Same runtime, contract and procedure as `robust_delay_v1_u150` (only `policy.onnx` differs): `--policy
stationary_hands_S2far2_u330`, first `g1.sh wb shadow ...` (motors untouched), then `active` on the gantry. The robot
image copies `cat/wholebody/` whole: rebuild or reload the wholebody image to get this folder on the robot.

## Robustness and status

Trained with the same randomisation as `robust_delay_v1_u150` (action delay 0-20 ms, SLAM-rate pose noise, LiDAR-mount
error, friction x0.5-1.2, mass x0.9-1.1, gain/torque/observation noise, pushes), starting from that policy. **Not yet
sim-to-sim tested and not evaluated under the training noise**: the numbers above are deterministic, noise-free
simulation, one seed. Start supervised.

## Provenance

`policy_contract.json` differs from `../policy_contract.json` only in `name`, `checkpoint_sha256`, `policy_onnx` and
`robustness`; `cat/tests/test_wholebody.py` checks that, the ONNX hash, and that the ONNX reproduces the training actor
on `reference_actions.npz`. Made with `cat/tests/ship_wholebody_variant.py` from the training repository's deploy bundle
`outputs/deploy/hands_S2far2_u330/` (Click-and-Traverse-Mjlab, branch feature/mjlab-migration); checkpoint
`outputs/saved_checkpoints/stationary_hands_S2far2_u330.pt`, sha256 `c543e4fd5bddd17a...` (full hash in
`policy_contract.json`). Results: `docs/stationary_hands_20261113.md`, "Hand retreat distance (S2far, S2far2)".
