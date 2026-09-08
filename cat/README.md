# CAT live 3D obstacle preview — NO MOTION

Branch: `feature/cat-perception-preview`. This is the **first perception
milestone**, not a walking-policy deployment. It does not start Nav2, consume
goals, publish velocity/joint commands, change FSM mode, arm/disarm the existing
gateway, broadcast TF, or attach CAT policy shared memory. No Unitree SDK or
upstream locomotion module is installed in this image.

## Start on this G1

Keep the robot stationary for SLAM initialization. Leave navigation disarmed;
do not run `nav-start-safe`, `g1-motion-arm`, or upstream CAT deployment scripts.
If navigation is already armed/running, use the existing `killnav` yourself
before this test. The preview is **not** a replacement for an emergency stop.

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
It has no restart policy. The launcher rejects an existing motion-arm token
and mounts the gateway's runtime directory **read-only** to continue checking
that token. If a token appears, visualization is invalidated/paused; this does
**not** stop some other controller or disarm it. An absent token also does not
prove that unrelated software is not controlling the robot.

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
| `/g1_cat/input_cloud` | Current nearby deskewed returns in map coordinates |
| `/g1_cat/ground_points` | Returns below floor + 10 cm, not obstacle hits |
| `/g1_cat/obstacle_points` | Current non-ground obstacle returns |
| `/g1_cat/occupancy_raw` | Actual occupied OctoMap cell centers, unmodified |
| `/g1_cat/occupancy_cat` | CAT morphology/fill result; compare with raw |
| `/g1_cat/roi` | Wire box: 5.12 × 5.12 × 1.40 m, bottom at floor z=0 |
| `/g1_cat/diagnostics` | JSON in a String; inspect in Raw Messages |

Expect `state=PREVIEW_OK`, advancing `sequence`, plausible nonzero
`obstacle_returns` and `occupied_raw`, and reasonable `processing_ms`.
`policy_ready` and `motion_enabled` are **always false**. PREVIEW_OK means
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
existing Livox driver -> existing FAST-LIO deskewed body cloud
  + existing sensor-time map -> world -> camera_init -> body TF
    -> map-frame returns, LiDAR-origin ray tracing, ground split
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
  the planar navigation-base frame. FAST-LIO has already applied LiDAR-to-IMU
  rotation. Ray origin uses the existing LiDAR translation in body:
  `[-0.011, -0.02329, 0.04412]` m.
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
- Preview runs at 1 Hz, samples at most 20,000 points per scan. This is not
  the 50 Hz control/observation loop. History resets every 10 seconds, on large
  pose/window changes, stale data, or a memory guard. This deliberately bounds
  memory/old obstacles during the experiment; it is not a persistent world map.
- TF is evaluated at sensor time; a latest-common-time fallback is allowed only
  within 0.25 s of the input scan, explicitly reported. Live receive age is
  independently checked. Missing/stale data clears output and reports invalid.
  Output clouds are already in map coordinates, stamped at publication for
  Foxglove. Source timestamps and clock offset remain visible in diagnostics.
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
network and imports the ROS node without starting it. GitHub Actions builds
and runs these tests natively on ARM64. For development **only**, if CMake,
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
