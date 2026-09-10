# CAT live 3D obstacle preview — NO MOTION

Optional next milestones are now available as `start-research`: potential-field
visualization, checkpoint verification and read-only G1 joint/IMU geometry.
See [RESEARCH.md](RESEARCH.md). No live policy inference or motor output is enabled.

Branch: `feature/cat-perception-preview`. This is the **first perception
milestone**, not a walking-policy deployment. It does not start Nav2, consume
goals, publish velocity/joint commands, change FSM mode, arm/disarm the existing
gateway, broadcast TF, or attach CAT policy shared memory. No Unitree SDK or
upstream locomotion module is installed in this image.

## Start on this G1

Keep the robot stationary for SLAM initialization. CAT perception can run
alongside either armed or disarmed navigation; starting/stopping CAT does not
change navigation's arm state. CAT does not feed these obstacles into Nav2 or
run a walking policy. Existing navigation remains the operator's responsibility;
the preview is **not** an obstacle-avoidance safety layer or emergency stop.
Do not run upstream CAT locomotion deployment scripts for this preview.

Use a **separate checkout**, never replace `~/g1_slam_deploy`:

```bash
git clone --branch feature/cat-perception-preview \
  https://github.com/Skvayzer/humanoid_navigation.git ~/humanoid_navigation_cat
cd ~/humanoid_navigation_cat
bash cat/scripts/build.sh
bash cat/scripts/container.sh validate

# Existing commands; normal slam-start localizes in teaching_lab_aligned.pcd.
slam-start
foxglove-start

bash cat/scripts/container.sh start
bash cat/scripts/container.sh logs
```

If the checkout or SLAM is already present/running, skip clone/start respectively.
`build.sh` requires sudo Docker access. It does not install anything on the host.
The image is `g1-cat-perception:preview`, container `g1-cat-perception-preview`.
It has no restart policy. CAT does not read or modify the motion-arm token and
does not mount the gateway's runtime directory. The previous disarmed-only
restriction was removed at the operator's request on 2026-09-10. No change was
made to the gateway, its watchdog/limits, Nav2, SLAM, or robot locomotion mode.

```bash
bash cat/scripts/container.sh status
bash cat/scripts/container.sh logs
bash cat/scripts/container.sh stop
```

Stop affects only the label-checked CAT container; it preserves the last log
and snapshot in `~/g1_cat_runtime` and leaves SLAM, Nav2, vendor drivers and
rosbridge alone. Restart with `start`. No existing shortcuts are overwritten.

## Foxglove test

Connect using **Rosbridge**, `ws://192.168.50.179:9090`. Fixed frame: **map**.
The existing bridge permits discovering `/g1_cat/*`; no bridge patch is needed.
Add these layers one at a time:

| Topic | Meaning |
| --- | --- |
| `/g1_cat/input_cloud` | Raw LiDAR returns aligned using point times and SLAM poses |
| `/g1_cat/nearfield_points` | Valid returns closer than 0.5 m to the LiDAR, inside the volume |
| `/g1_cat/ground_points` | Returns below floor + 10 cm, not obstacle hits |
| `/g1_cat/obstacle_points` | Current non-ground obstacle returns |
| `/g1_cat/occupancy_raw` | Actual occupied OctoMap cell centers, unmodified |
| `/g1_cat/occupancy_cat` | CAT morphology/fill result; compare with raw |
| `/g1_cat/roi` | Wire box: 5.12 × 5.12 × 1.40 m, bottom at floor z=0 |
| `/g1_cat/diagnostics` | JSON in a String; inspect in Raw Messages |

Expect `state=PREVIEW_OK`, advancing `sequence`, plausible nonzero
`obstacle_returns` and `occupied_raw`, and reasonable `processing_ms`.
`policy_ready` and `motion_enabled` are **always false for CAT**, not assertions
that the robot or Nav2 is disarmed; `navigation_arm_state=not_monitored` makes
that distinction explicit. PREVIEW_OK means
data processing succeeded, not that localization/ground height is correct.

With the robot stationary and disarmed, place/move an inanimate box, then
remove it; compare raw returns, occupied cells and cleared space. Check a low
obstacle and an elevated surface/overhang, keeping people clear of the robot.
Cells outside the LiDAR field of view remain unknown; removing an obstacle
doesn't clear it until new rays pass through it (or bounded history expires).

**Reject this test** if the box is flipped, floor height is wrong, nearby
obstacles are missing, pose jumps, or scans/TF are stale. Do not fix it with
another arbitrary rotation: verify existing localization and calibration.
Do not arm anything based on this preview.

## Data flow and differences from upstream

```text
existing /livox/lidar CustomMsg (raw points and point timestamps)
  -> assemble packet data into 100 ms sensor-time windows
  + existing SLAM sensor-time map -> world -> camera_init -> body TF
    -> bounded scan queue; newest scan fully covered by TF
      -> calibrated raw-LiDAR-to-body transform + 10 ms point-time pose bins
        -> map-frame returns, moving-LiDAR ray origins, ground split
      -> OctoMap (.04 m; hit .7 / miss .4; clamp .12 .. .97)
        -> bounded XYZ grid [128,128,35]
          -> upstream CAT morphology / vertical fills
            -> visualization and offline snapshot ONLY
```

CAT commit/provenance: [upstream/NOTICE.md](upstream/NOTICE.md). CAT is a
3D learned locomotion policy, not a drop-in Nav2 `cmd_vel` controller. The
existing SLAM/Nav2 source and configuration are unchanged on this branch.

Adapter details and intentional limitations:

- Uses our already floor-aligned `map` instead of CAT's separate RANSAC
  `floor_init`. Does **not** rerun its driver/SLAM, flip gravity again, or use
  the planar navigation-base frame. The raw cloud has NOT had FAST-LIO's
  extrinsics applied, so CAT applies the same `diag(1,-1,-1)` rotation and
  `[-0.011, -0.02329, 0.04412]` m translation exactly once. This corrects the
  G1 vendor cloud/IMU axis mismatch without using the legacy visualization TF.
- CAT's `min_range: 0.0` means **no software blind-distance cutoff**. Zero-length,
  nonfinite, invalid-tag/line and malformed timestamp returns are still rejected.
  Zero returns never clear rays either. Close returns (<0.5 m) take precedence
  over farther points if sampling is needed. This does not overcome hardware
  minimum range, occlusion, or the LiDAR's field of view. FAST-LIO and its
  registered-cloud topics retain their tested `blind: 0.5`; only CAT bypasses
  that filter by reading the vendor stream directly. Robot-body returns may
  now be visible: no validated body mask has been guessed or silently applied.
- CAT's layout is XYZ, map-axis aligned, 4 cm cells, floor-relative Z 0..1.4 m.
  The XY window follows the current body position and snaps to voxel boundaries.
  Heights above 1.4 m are **not represented**. Below-10-cm obstacles are treated
  as ground, matching the upstream ground-filter threshold; this is not a
  validated floor/step detector. The crane and robot's own body may be included;
  no validated self-collision/body masking has been added yet.
- Maximum integration range 2.5 m. A return beyond range clears its truncated
  ray but does not create a hit there. Ground rays clear observed space without
  making ground occupied. Within a scan, occupied endpoints win over free rays.
- Uses native OctoMap, but a bounded extraction wrapper instead of upstream's
  full 512-cubed dense conversion. Unknown=0, free=1, occupied=2 are retained
  separately. The CAT binary mask alone **cannot distinguish unknown from free**.
  It is not yet safe input to a moving policy.
- Exact CAT closing (5³, two iterations), neighbor test, upward fill from
  Z-index 20 and downward fill below 5. These operations can erase thin/boundary
  obstacles and invent filled regions. Diagnostics report additions/removals.
  No fake corner obstacles are added on empty input.
- Preview runs at 1 Hz with at most 20,000 valid returns. The reader uses a
  NumPy view of validated Livox CDR1 data, avoiding per-point
  Python ROS-object creation that can starve TF reception on Foxy. The decoder
  is checked against ROS serialization, endian/alignment and truncated data.
  One bounded processing worker runs independently of the ROS input/TF receive
  loop (no concurrent overlapping map updates). Both large CustomMsgs and small
  96-point packets are assembled into 100 ms sensor-time windows. Absolute point
  timestamps are preserved when offsets are rebased, including unsorted offsets
  and packets that cross a window boundary. A later packet's start timestamp
  closes earlier windows; receive-time silence never fabricates a complete scan.
  Input gaps over 25 ms discard partial windows, and windows need at least 75 ms
  of point-time coverage. DDS loss can still reduce density; batching does not
  recreate missing points. Duplicate packets do not refresh input age.
  History is retained by the oldest constituent packet's monotonic receive time
  (`max_receive_age`, 1.5 seconds), not a fixed raw-message count. Additional
  limits of 500,000 buffered points (~10 MB point data plus metadata), 64 completed
  scans and 2,048 pending pieces bound memory. Budget/clock resets clear both
  partial and completed history. The newest fresh complete scan covered by TF
  is selected. There is no latest-pose
  fallback: each 10 ms point-time bin requires its own full map-to-body pose.
  This uses interpolation of SLAM poses, not FAST-LIO's IMU-integrated deskewer;
  it is an approximation that still requires testing under movement.
- No periodic whole-map reset. Observed cells persist and new free rays clear
  obstacles probabilistically. Cells not observed for 30 seconds or outside the
  rolling window become **unknown**, not free. Native storage is compacted while
  preserving retained cells' probabilities. This avoids an OctoMap 1.9.3
  per-node-deletion assertion without modifying any host library. Major pose
  jumps, sensor-clock resets, long data loss and memory faults still clear history.
- Before publishing a new result, input age must be below 2 seconds. Short data
  gaps hold the last visualization, with **amber ROI**, `HOLDING_STALE` and
  `data_valid=false`; cloud timestamps are NOT refreshed to disguise stale data.
  After 3 seconds since the last accepted input was received, output/history
  clear. Navigation arming does not invalidate CAT output. Only `PREVIEW_OK` sets
  `data_valid=true`; `policy_ready` remains false in all states. No stale held
  visualization should ever be connected as an actionable policy observation.
- Foxy's Python TF listener uses relative topic names. The CAT node explicitly
  remaps its TF inputs to `/tf` and `/tf_static`; `/g1_cat/tf*` are not inputs
  and must not be populated with fabricated transforms to work around an error.
- Docker shares networking for vendor discovery, but no devices, host IPC/PID,
  privileged capability, writable existing workspace, SDK or motor publisher.
  Read-only root filesystem, 2 CPU and 1.5 GiB limits. Network sharing is **not**
  DDS security enforcement: the no-motion property comes from this audited
  perception-only code/image, not a firewall prohibition on command topics.

## Offline snapshot and tests

Every five successful updates, `~/g1_cat_runtime/latest.npz` is replaced with
the last test sample: `occupancy_state`, `cat_occupancy`, `origin`, `resolution`,
`source_stamp`, and JSON `metadata`. It is an **offline** sample, not a live
policy buffer; after disconnect/stop it remains the last recorded sample.

```python
import numpy as np
sample = np.load('latest.npz', allow_pickle=False)
print(sample['occupancy_state'].shape)  # (128, 128, 35), unknown/free/occupied
print(sample['origin'], sample['resolution'], sample['source_stamp'])
```

`container.sh validate` runs native/geometry/no-motion tests without a robot
network. It also constructs the actual ROS node in isolated test domain 101
and verifies reception of a synthetic global transform with its processing
timer disabled. This catches namespaced-TF wiring mistakes that import/unit
checks alone miss. GitHub Actions builds and tests natively on ARM64.
For development **only**, if CMake,
G++, liboctomap-dev, NumPy and SciPy are already installed, use
`bash cat/scripts/build_native.sh`; it writes only `cat/build` and starts no ROS
node. Do not install the upstream training environment on the robot.

## Gates before a later policy preview

1. Verify live frame/floor alignment, near/far/overhead obstacles, clearing,
   stale-data handling and CPU/memory alongside SLAM.
2. Add CAT's SDF/boundary/guidance fields with bounded computation, explicit
   unknown-space policy and goal-frame conversion. Compare against pinned
   upstream fixtures, not just plausible-looking visuals.
3. Verify G1 embodiment (joint ordering, posture, limits, proprioception,
   body-frame conventions), obtain/checkpoint-hash the appropriate policy,
   then implement inference **without any actuator publisher**.
4. Log/inspect shadow inference. A future real-motion stage needs a separate
   review and explicit operator approval. Never run upstream `deploy_real_gf.py`
   or its low-level controller as a supposed disarmed test: even upstream debug
   paths can publish low-level commands.

## Updating the preview

After source/config updates, rebuild and replace **only** CAT:

```bash
cd ~/humanoid_navigation_cat
git pull --ff-only
bash cat/scripts/container.sh stop
bash cat/scripts/build.sh
bash cat/scripts/container.sh validate
bash cat/scripts/container.sh start
bash cat/scripts/container.sh logs
```

SLAM/localization and rosbridge stay running. Inspect `/g1_cat/nearfield_points`
and compare `nearfield_available`, `nearfield_retained`, `nearfield_in_volume`
with `/g1_cat/occupancy_raw`; CAT morphology and the unchanged height cutoff
can still remove points from the final processed obstacle grid.

`input_buffer` diagnostics report received packet count, assembled/queued scans,
buffered points, discarded partial windows and memory-budget resets.
`batch_packets` and `batch_duration_ms` identify the actual scan being processed.
If Docker log capture fails during `stop`, a warning and any partial log are
saved and the helper still removes only the stopped, label-checked CAT container.
