# Perception-only validation notes

## 2026-09-10 — research fields, checkpoint audit, read-only telemetry

Implementation commits: `236c396`, `5fa5758`. ARM64 workflow run
`34481366762` passed: image build, 75 in-container tests (two host-only launcher
cases skipped there and passed on the CI host), all seven repository checks,
and pinned ONNX artifact/config verification plus synthetic inference.

Coverage includes SDF/gradient/guidance parity with pinned CAT reference
samples, unknown/occupied/out-of-volume goal rejection, unreachable-region
masking, voxel-center sampling, FK parity with three MuJoCo fixtures, telemetry
validation/freshness/tick reset, research goal/field invalidation and actual
LowState reception/publication serialization in isolated ROS domain 101.
The first CI run exposed old SciPy's missing `Rotation.as_matrix`; FK now uses
the existing tested quaternion routine and passes on Foxy's stock SciPy.

A five-second live **read-only** probe on this G1 accepted 675 advancing
LowState samples, ticks 263152..268148, with no telemetry/FK-limit errors.
Only the three telemetry message definitions were built in
`/tmp/g1-cat-research.EsaJEt`. The probe had no actuator publisher, did not
change services/modes/arming, and did not import the robot SDK. This verifies
wire reception and model-limit consistency, not physical embodiment or
map-to-pelvis calibration.

An offline copy of the existing CAT snapshot yielded 305,067 observed-free
cells (8,898 in the displayed height slice); a synthetic goal chosen only in
the offline test yielded 304,991 reachable cells and 304,960 nonzero vectors.
No goal was published to any ROS topic. Desktop processing was approximately
207 ms distance-only / 365 ms with guidance; these are not G1 timings or
evidence of real-time walking readiness. The snapshot was not committed.

The current G1 CAT image has not been rebuilt/replaced by this work: SSH sudo
Docker access needs the operator's password. `start-research` and the end-to-end
Foxglove check are the remaining deployment steps in `RESEARCH.md`. No SLAM,
Nav2, driver, bridge, gateway, firmware or robot-mode settings were changed.
Live policy inference and actuation remain unimplemented and disabled.

## 2026-09-10 — packet-size independent scan buffering

The live driver was publishing 96-point CustomMsgs at roughly 1.5–1.7 kHz,
whereas the previous successful run logged about 20,000 points spanning 100 ms
per message. The launch file still requested `publish_freq: 10.0`; its batching
change was not attributed to a confirmed setting change. The CAT adapter's
16-message history held only milliseconds, so no queued input survived until
the matching map/body TF arrived 0.2–0.6 seconds later.

The fix is entirely in CAT: 100 ms point-time assembly, time-based history
retention, bounded point/object storage, conservative oldest-receive freshness,
and unchanged exact sensor-time TF requirements. No upstream driver, Nav2,
SLAM, extrinsics, floor, controller, or arming changes. The launcher also now
continues removal of the stopped CAT container if Docker log capture fails.

All 51 CAT tests passed in the robot's temporary test directory, including
15 packet-assembly regressions, native occupancy tests, no-motion checks, and
actual TF reception in isolated ROS domain 101. All seven repository source
verification tests also passed. These checks did not modify live services.

A 36-second passive run against real LiDAR/TF produced 35 PREVIEW_OK updates,
all 35 occupied clouds nonempty. Samples contained 5,300–18,558 input points
across 56–194 received packet pieces and 94–100 ms of sensor time. Processing
including local ROS serialization took approximately 0.45–0.75 seconds/update;
output receive ages stayed below 1.37 seconds. The final buffer retained 13
scans spanning 1.35 seconds, ~194k points, and reported zero budget/clock resets.
Packet loss/receive gaps still reduce density; the adapter does not fabricate
missing returns. The ROS receive depth and all freshness limits are unchanged.

The passive probe replaced every CAT visualization publisher with a local
serialization sink. It sent zero visualization/TF/motion messages, did not
change running services, and did not use the robot SDK. It ran from a temporary
host directory alongside the existing processes, not under Docker's CPU quota.
The running CAT image must still be rebuilt/replaced by the operator's sudo
session, followed by an end-to-end Foxglove check.

## 2026-09-10 — allow passive CAT alongside armed navigation

Operator explicitly requested keeping navigation armed while running CAT.
Removed the launcher and both runtime arm-token checks, the token parameter,
and the navigation-runtime bind mount. CAT neither monitors nor modifies the
gateway's arm state. Diagnostics now explicitly say
`navigation_arm_state=not_monitored`; `motion_enabled=false` describes CAT only.

The five node gate tests, launcher test (fake Docker), and ten stream/decoder
tests passed using the robot's ROS environment in a temporary directory.
The launcher regression verifies startup with a test arm token present; the
node regression completes processing/publication without reading or changing
that token. No live ROS processing or robot-control calls were made by these
tests. All seven repository verification tests also passed.

Navigation's actual arm token was present when checked. No change was made to
Nav2, SLAM, the motion gateway, watchdogs, velocity limits or locomotion mode.
The operator must rebuild and start only CAT to use the updated runtime. This
does not connect CAT obstacles to Nav2 or activate the CAT locomotion policy.

## 2026-09-08 — occupancy continuity and close-range input

Progress checkpoint: `5ee310e` (intentionally marked WIP; not deployed).

Problems reproduced on the G1:

- Brief scan/TF skew made the previous preview publish empty clouds.
- Whole-OctoMap reset every 10 seconds discarded accumulated observations.
- FAST-LIO registered clouds had already removed returns closer than 0.5 m.
- A naive Python CustomMsg subscription created thousands of point objects,
  causing its TF receive queue to fall behind. A worker thread alone did not
  fix that. The raw CDR1/NumPy reader removes that allocation bottleneck.

Implemented exclusively under `cat/`:

- Short gaps retain the last visualization with invalid/stale diagnostics;
  sustained data loss still clears output. No fabricated TF fallback.
- Per-cell observation age and rolling bounds replace periodic whole-map reset.
- Raw `/livox/lidar` input with `min_range: 0.0`, validated return tags and times,
  zero-return rejection, calibrated extrinsics, and 10 ms pose-time bins.
- Close returns are prioritized before sampling and shown in `nearfield_points`.
- Bounded processing worker and direct native occupancy export.

Checks completed before deployment:

- 35 CAT tests passed on robot ARM64, including actual ROS serialization versus
  the decoder, endian/alignment/truncation checks, global TF reception in an
  isolated ROS domain, cell expiry/compaction, near-range returns, and no-motion
  source checks.
- All 7 repository verification tests passed, including preservation of the
  original SLAM/navigation algorithms and configuration.
- Final passive 36-second test: 36 PREVIEW_OK updates, all 36 occupied clouds
  nonempty, with history continuing past both 10 and 30 seconds. Processing
  took approximately 0.48–0.59 seconds/update. Observed 168–243 near-field
  returns per update inside the volume (returns may include robot/crane).
- Probes subscribed to real scans and TF but replaced cloud/marker/status
  publication with local inspection. No motion commands, TF, goals or changes
  to existing processes were sent. The motion-arm token remained absent.

These timing measurements were taken on the host in a temporary test directory,
alongside the existing SLAM/localization/rosbridge/old CAT container. They are
not a measurement under the new container's CPU quota or Foxglove transport.
The operator must rebuild/restart only CAT, then verify `PREVIEW_OK`, near-field
geometry and stable visualization. Docker sudo access was not available to the
remote development session. The previous running image was left untouched.

Still unvalidated: moving-robot alignment, moving-obstacle clearing, body
masking, hardware blind areas, and policy usability. This remains visualization
only; `policy_ready=false` and `motion_enabled=false` in every state.
