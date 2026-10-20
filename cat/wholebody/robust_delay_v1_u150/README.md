# generalist_robust_delay_v1_u150 — action-delay and perturbation resistant

The shipped policy (`../policy.onnx`, generalist_finetune_v1_u596) fine-tuned for 150 updates with action latency
0-20 ms, pelvis-pose noise held at the SLAM rate, LiDAR-mount error, foot friction x0.5-1.2 and link mass x0.9-1.1
(plus the original gain, torque and observation noise). Same 226 -> 29 interface and the **same runtime contract**
(`policy_contract.json` differs from `../policy_contract.json` only in name, hashes and the `robustness` record), so
`wholebody_core.py` runs it unchanged: load this `policy.onnx` instead of `../policy.onnx`.

Sim-to-sim (plain MuJoCo, physical obstacles; flat 4 m, 0.47 m gap, 0.48 m corridor, 0.44 m gap; 5 seeds each):

| Condition | parent (u596) | this policy |
|---|---|---|
| action delay 0 / 10 / 20 ms alone | 20/20, 20/20, 20/20 with 2 falls | 20/20, 20/20, 20/20, **0 falls** |
| action delay 40 ms | 20/20 fall | 20/20 fall |
| full 20 ms delay + per-step pose noise | 14/20 reached, 9 falls | 16/20 reached, 5 falls |
| worst case (above + 40 ms pose latency, kp -10%, perception limits) | 13/20 reached, 10 falls | 19/20 reached, 2 falls |
| robot-like (pose noise at 10 Hz, 40 ms pose latency, 15 ms action latency, kp -10%) | 20/20 | 20/20 |

No obstacle contacts in any run. Limit: keep observation-to-motor delay below ~20 ms; stop the policy on a loop
overrun (it falls at 40 ms).

Full deploy bundle (deploy_spec, reference rollouts, TorchScript): training repository
`outputs/deploy/generalist_robust_delay_v1_u150/`; checkpoint `outputs/saved_checkpoints/generalist_robust_delay_v1_u150.pt`
(sha256 in `policy_contract.json`).
