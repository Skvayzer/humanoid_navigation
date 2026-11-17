# stationary_hands_T2_u410 — table passages with raised hands

Walks between two tables facing forward and lifts its hands above the table tops, instead of turning sideways.
Lineage: `robust_delay_v1_u150` → S2 → S2far u200 → T1 → T1b → T1c (table passages: facing cost, leg power, hands-up
band) → T2 (certified table-gap width curriculum 0.46-0.70 m), 410 updates.

## What it does (simulation)

| Table pair, gap | goal reached, facing forward |
|---|---|
| 0.62-0.70 m | 99% (turned sideways near the tables 4%; 53% before the width curriculum) |
| 0.54-0.62 m | 95% |
| 0.46-0.54 m | 51% |
| 0.44 m | 2% (geometric limit: about 4 cm per side) |

Traversal kept: 0.59 clean goal-reaching on 384 non-table scenes.

## Limits

* Stops in front of narrow cabinet passages (< 0.60 m) far more often than before the table training.
* The 0.44 m pairs are beyond it; trials from 0.70 m down to 0.55 m, with padded table edges.

## Run

Same runtime, contract and procedure as `robust_delay_v1_u150` (only `policy.onnx` differs): `--policy stationary_hands_T2_u410` on
the executor, e.g. `g1.sh wb shadow --policy stationary_hands_T2_u410 --scene <scene> --anchor-at-robot` first (motors untouched), then
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
from the training repository's deploy bundle `outputs/deploy/hands_T2_u410/` (Click-and-Traverse-Mjlab, branch
feature/mjlab-migration; full bundle with deploy_spec, rollouts and TorchScript); checkpoint
`outputs/saved_checkpoints/stationary_hands_T2_u410.pt`, sha256 `966c5892d767230a3cca920b371aa4ff19999f5f582003620b43399bacffda66`. Training log: `docs/stationary_hands_20261113.md`.
