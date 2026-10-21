# generalist_robust_delay_v2_u552: action-delay and perturbation resistant (real-deployment randomisation)

Continuation of the robust v1 line (robust v1 final state, update 226) for 552 updates with the randomisation matched
to one proven G1 recipe, KungfuBot / PBHC (released HumanoidVerse config): **action latency 0-40 ms** (motor side),
**torso centre-of-mass offset +-5 / +-5 / +-1 cm**, **foot friction 0.2-1.2**, **constant per-episode torque offset
+-5% of the limit**, plus everything v1 had (mass x0.9-1.1, kp/kd x0.75-1.25, random torque 10%, pushes, observation
noise, SLAM-rate pose noise, LiDAR-mount error). Same 226 -> 29 interface and the **same runtime contract** as
`../policy_contract.json` (only name, hashes and the `robustness` record differ): load this `policy.onnx` instead.

Sim-to-sim (plain MuJoCo, physical obstacles; flat 4 m, 0.47 m gap, 0.48 m corridor, 0.44 m gap; 5 seeds each):

| Condition | robust_delay_v1_u150 | this policy |
|---|---|---|
| motor-side latency 0 / 10 / 20 ms | 0 falls | 0 falls |
| motor-side latency 30 / 36 / 40 ms | 0 / 0 / 2 falls | **0 / 0 / 0 falls** |
| 40 ms + pose noise 1 cm + 40 ms pose latency + kp -10% + perception limits | 4/20 falls | **0/20 falls** |
| runtime one step behind (`--action-delay 1`) + pose noise | 5/20 falls | **0/20 falls** |
| same + pose latency, kp -10%, perception limits | 2/20 falls | 1/20 falls |
| robot-like (pose noise at 10 Hz, 40 ms pose latency, 15 ms latency, kp -10%) | 0 falls | 0 falls |

No obstacle contacts in any run. The runtime must apply the newest action every step (two steps behind, both fall).

**Skill retention vs v1 (hurdles, crouch, gaps by width, rooms) is being measured** under identical nominal conditions
(training repo `outputs/robust_v2_skill_eval/`); the training-time success rates are lower, but under the harder
randomisation, so not comparable. Check that comparison before choosing this policy over v1 for obstacle courses.

Full bundle: training repository `outputs/deploy/generalist_robust_delay_v2_u552/`; checkpoint
`outputs/saved_checkpoints/generalist_robust_delay_v2_u552.pt` (sha256 in `policy_contract.json`).
