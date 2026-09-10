# CAT next milestones: fields, checkpoint audit, read-only robot state

Branch: `feature/cat-perception-preview`. This extends, not replaces, the
working perception preview. **No real-state policy inference or robot actuation
is implemented or enabled.** No motor interface is generated in the CAT image.
The container does not import the Unitree SDK, open policy shared memory,
publish global TF, call robot services, consume Nav2 goals or change arm state.
There is no motion mode. Existing navigation remains independently controlled.

## Implemented

1. Pinned Generalist v1 artifact download, hash/size/config verification and an
   optional offline ONNX graph/synthetic-input smoke test. Binary model files
   are ignored by Git and excluded from the Docker context. Runtime nodes do
   not load a checkpoint; `deployment_approved` is always false.
2. CAT SDF, boundary gradient and fast-marching goal-guidance preview. Arithmetic
   is regression-tested against the pinned upstream implementation in known
   space. Unseen cells block guidance; observed thin obstacles removed by CAT
   morphology are retained. These conservative differences are intentional.
   Unreachable cells have zero guidance. Empty known scenes need no invented
   corner obstacles. Entirely unknown scenes are rejected.
3. Read-only `/lowstate` subscription, validated 29-joint/IMU extraction,
   telemetry freshness, and upstream G1 forward kinematics. FK is compared
   against MuJoCo-generated fixtures. Only telemetry message definitions are
   bundled. No SDK or controller module is needed.

## Start on the G1

Keep the existing SLAM/localization and visualization bridge running. Do not
start the upstream CAT deployment launcher. Build and validate the new image
before replacing the existing CAT container:

```bash
cd ~/humanoid_navigation_cat
git pull --ff-only
bash cat/scripts/build.sh && bash cat/scripts/container.sh validate

# If the old CAT container exists:
bash cat/scripts/container.sh stop

bash cat/scripts/container.sh start-research
bash cat/scripts/container.sh logs
```

`start-research` uses the same single CAT container name, resource limits and
read-only mounts. It must not coexist with another copy of this CAT preview.
The original `start` still selects perception-only mode. To return to it, stop
CAT and use `start`. Neither command starts/stops Nav2 or SLAM or reads/writes
navigation's arm token. Replacing CAT does not make armed navigation safer.

## Foxglove

Keep the existing `map` 3D panel and add:

| Topic | Meaning |
|---|---|
| `/g1_cat/sdf_slice` | Observed-free XY samples near floor + 0.75 m; intensity = signed distance in metres to occupied/unknown boundary |
| `/g1_cat/boundary_preview` | Orange directions away from occupied/unknown regions |
| `/g1_cat/guidance_preview` | Cyan directions toward the dedicated preview goal; requires a valid goal |
| `/g1_cat/field_diagnostics` | Field freshness, sequence, coverage, goal/rejection reason |
| `/g1_cat/robot_state_diagnostics` | LowState freshness, IMU values, calibration/embodiment gates |
| `/g1_cat/joint_states_preview` | Candidate 29-joint mapping; values only, not commands |

Create a **separate** 3D panel with fixed frame `g1_cat/pelvis_preview` for
`/g1_cat/body_sites_preview` and `/g1_cat/pelvis_gravity_preview`. This frame is
deliberately NOT connected to `map` by a guessed transform. It shows the model
in a pelvis-local view, not the robot's verified pose in the lab. Site order is
reported in `robot_state_diagnostics`. Joint/IMU receipt times are used because
LowState has no ROS timestamp; tick advances and receive age are checked.

Use Foxglove's Publish panel for a `geometry_msgs/msg/PoseStamped` on
**`/g1_cat/goal_preview`**, with `header.frame_id: map`, desired nearby XY and a
valid quaternion (identity is fine). This topic is not a Nav2 input. Z and yaw
are not commands: the 2D click seeds the field at floor + 0.75 m. Goals expire
after 60 seconds. A goal outside the volume or in occupied/unknown space is
rejected, not silently projected to somewhere else. With no goal, expect
`DISTANCE_ONLY_NO_GOAL`. With a valid goal, expect `FIELDS_PREVIEW_OK`.

The fields are computed in one bounded additional worker, at most every two
seconds; LiDAR/TF reception and existing perception continue independently.
No overlapping field tasks or unbounded queue. The original 2 s input-age
deadline applies before and after field computation. Invalid/stale perception,
goal changes and field timeout clear the field layers immediately; old worker
results cannot overwrite a newer goal or invalidate a newer field. Short field
visibility gaps are possible at this preview rate and are explicitly invalid,
not actionable policy observations. Body telemetry expires after 0.25 s.

Sparse known-free cells or a rejected goal may legitimately produce few/no
arrows. Do not disable unknown-space checks merely to obtain a prettier field.
The numerical .6 magnitude retained from upstream is **not a robot speed
command**. Vector lengths in Foxglove are normalized for inspection.

## Checkpoint audit

No host package installation is needed to fetch/verify artifacts:

```bash
python3 cat/checkpoint.py fetch
python3 cat/checkpoint.py verify
```

The manifest pins Hugging Face revision
`46ce4b57ba0639168d51741b661ff62f7ce6f045`, model/config sizes and SHA-256 hashes.
Different existing files are never overwritten. The selected model has a
dynamic batch dimension with 162 float observations and 12 action outputs.
Its config requests `ctrl_dt=.02`, incremental action scale `.5`. The ONNX
file's numerical output is **not** a velocity vector. Older deployment wrappers
must not be assumed to match it solely because tensor widths agree.

On a development machine, an isolated environment containing NumPy, ONNX and
ONNX Runtime can additionally run `python cat/checkpoint.py smoke`. CI does this
on ARM64. It checks the graph and executes 25 synthetic zero-input calls, never
imports ROS or consumes robot data. The resulting latency is a synthetic model
benchmark, not a measurement of the whole control loop or proof of safe actions.
The live CAT image deliberately contains no ONNX Runtime dependency.

## Still gated before shadow inference

- Confirm the physical G1 variant, motor indexing and sensor conventions.
- Measure/verify the calibrated transform from FAST-LIO's IMU/body reference to
  the appropriate robot link. Apply waist kinematics rather than assuming that
  sensor/body equals pelvis. No map-aligned body sampling is enabled until then.
- Verify self-body masking, close/low/overhead obstacles, pose discontinuities,
  unknown-space handling and moving-scan deskew. Current floor cutoff is 10 cm
  and volume ceiling is 1.4 m; this is not a full-body clearance guarantee.
- Match all 162 observation entries, preprocessing, phases, previous actions,
  incrementally integrated targets and joint limits against Generalist v1's
  training/evaluation implementation. Then add a no-actuator shadow runner.
- Benchmark end-to-end field and inference latency alongside SLAM under the
  container CPU quota. Visualization settings are not walking settings.

`policy_ready=false`, `inference_enabled=false`, `embodiment_verified=false`
and `map_alignment_verified=false` are intentional, not faults to bypass.
Real motor control remains a separate approval/review stage. The current
`killnav` is not a future CAT low-level stop mechanism.
