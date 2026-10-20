# g1_cat_bridge — CAT perception (ESDF) preview on our G1, no motion

> On branch `feature/g1-real-robot` this module lives in `real_robot/cat_bridge` and uses the checkout it is in
> (no separate `humanoid_navigation_cat` copy needed). Two ways to run it:
> * `perception.sh start` — robot standing still, pose measured once (stationary_pose), as described below;
> * `perception.sh start-with-slam` — while walking: start `real_robot/scripts/g1.sh slam odom|localize <map>` first;
>   the bridge then only converts the LiDAR stream and CAT takes `map -> body` from the SLAM.
> The two pose sources must never run together (both define `body` in TF); the scripts refuse that.

Runs the perception part of [humanoid_navigation `feature/cat-perception-preview`](https://github.com/Skvayzer/humanoid_navigation/tree/feature/cat-perception-preview)
on the G1 Jetson and shows it in Foxglove on the laptop. Nothing here commands the
robot: no SDK, no velocity/joint publishers, no mode changes, the LiDAR is not reconfigured.

```text
vendor main board (192.168.123.161)
  /utlidar/cloud_livox_mid360 + /utlidar/imu_livox_mid360      (read-only, best effort)
      └─ g1-cat-bridge container (this module)
           utlidar_to_livox → /livox/lidar (Livox CustomMsg) + /livox/imu
           stationary_pose  → /tf map→body (measured once, robot standing still)
               └─ g1-cat-perception-preview container (repo, unmodified, research mode)
                    → /g1_cat/* : occupancy, ESDF slice, boundary field, diagnostics
                        └─ g1-cat-rosbridge container → robot localhost:9090
                             └─ SSH tunnel → laptop ws://localhost:9090 → Foxglove
```

## Why the bridge exists

The repo was captured on a G1 whose Jetson ran its own Livox driver (`/livox/lidar`,
CustomMsg) and a Fast-LIO + NDT SLAM stack that provides `map → body`. On our robot:

- **LiDAR topic.** The MID360 is consumed by the vendor main board and republished as
  `/utlidar/cloud_livox_mid360` (PointCloud2: x, y, z, intensity, ring, time = ns since
  stamp). Running a second Livox driver would steal the LiDAR from the vendor stack,
  so `utlidar_to_livox` converts the vendor topic losslessly instead.
- **Axes.** The vendor cloud is in the native upside-down sensor axes (99.8% of
  returns lie in the MID360's native −7°…+52° elevation band). The repo's driver
  published it rolled 180° about X and both Fast-LIO and CAT undo that once with
  `diag(1,-1,-1)`. The converter applies that roll so the repo code is unchanged.
  IMU is passed through unchanged (native axes, g units), matching the repo.
- **Pose without SLAM.** For a robot that is standing still, `map → body` is constant.
  `stationary_pose` measures it: tilt from IMU gravity, height from a least-squares
  floor-plane fit (on our robot: IMU 1.242 m above the floor, tilt 5.7°, floor fit
  RMS 1.1 cm). The map origin is under the IMU, map X = projected body X, floor z = 0.
  If the robot rotates or tilts by more than 2°, it stops publishing and CAT goes stale
  (`HOLDING_STALE`, then empty) instead of drawing a wrong map. Restart after that.

This is valid **only while the robot stands still**. Walking requires the repo's SLAM.

## Robot layout (`~/workspace/artemvoronov`)

| Path | Contents |
|---|---|
| `humanoid_navigation_cat/` | repo checkout, unmodified (+ verified ARM64 runtime libs in `vendor/`) |
| `g1_cat_bridge/` | this module (copied from the laptop) |
| `runtime/cat/` | CAT snapshots (`latest.npz`) and last container log |
| `runtime/slam/` | repo SLAM maps and logs (via `config/site.env`) |
| `logs/` | build logs, last bridge log |

Docker images (stored by Docker in `/var/lib/docker`): `g1-cat-perception:preview`,
`g1-cat-bridge:latest`, `g1-cat-rosbridge:latest`, `g1-slam-foxy:backup-20260713` (repo SLAM,
not needed for the stationary ESDF preview), base `ros:foxy-ros-base-focal`.

`humanoid_navigation_cat/config/site.env` (gitignored) points the repo's SLAM runtime
(`G1_SLAM_RUNTIME_DIR`) to `runtime/slam/` so nothing is written to `~/g1_slam_runtime`.

**After a robot reboot** just run `perception.sh start` again; it removes our stopped
containers first. A reboot also restores the dead eth0 default route (no internet), which
only matters for rebuilding.

## Run it

1. **Robot:** powered, standing still (crane or standing mode), nothing moving near it during the first ~5 s.
2. **Start on the robot** (asks for the robot's sudo password):
   ```bash
   ssh g1-wifi                     # or: ssh g1  (Ethernet cable, 192.168.123.164)
   cd ~/workspace/artemvoronov
   bash g1_cat_bridge/scripts/perception.sh start
   bash g1_cat_bridge/scripts/perception.sh status
   ```
   `start` waits for `Pose fixed: IMU height above floor …`, then starts CAT
   (research mode) and rosbridge.
3. **Tunnel on the laptop** (keep this terminal open):
   ```bash
   ssh -N -L 127.0.0.1:9090:127.0.0.1:9090 g1-wifi
   ```
4. **Foxglove:** open <https://app.foxglove.dev> in Chrome (free sign-in) or the Foxglove
   desktop app → *Open connection* → **Rosbridge** → `ws://localhost:9090`.
   Then *Layouts → Import from file* → `foxglove/g1_cat_esdf_layout.json` from this module.
5. **Stop** (robot): `bash g1_cat_bridge/scripts/perception.sh stop`.

### What to look at (fixed frame `map`)

| Topic | Meaning |
|---|---|
| `/g1_cat/sdf_slice` | **ESDF slice** at floor + 0.75 m: observed-free cells, intensity = signed distance (m) to the nearest occupied/unknown boundary |
| `/g1_cat/boundary_preview` | arrows pointing away from obstacles/unknown space (normalized length) |
| `/g1_cat/occupancy_cat` / `occupancy_raw` | CAT-processed vs raw OctoMap cells (4 cm), 5.12 × 5.12 × 1.4 m ROI |
| `/g1_cat/obstacle_points`, `ground_points`, `input_cloud` | split LiDAR returns |
| `/g1_cat/roi` | ROI box (amber = stale data) |
| `/g1_cat/field_diagnostics` | expect `DISTANCE_ONLY_NO_GOAL` (no goal set), advancing `sequence` |
| `/g1_cat/diagnostics` | expect `state: PREVIEW_OK`, `data_valid: true` |

**Why the ESDF blinks without decay time.** On the Jetson, perception (~0.9 s) plus the field
computation (~0.9 s) make each ESDF ~1.7 s old when published. CAT's freshness rule
(`max_output_age: 2.0` s) then clears it ~0.3 s later and publishes an empty slice
(`field input expired; old arrows removed`) until the next field ~2 s later. This is the repo's
intended safety behaviour, not a fault. The layout sets **Decay time = 2.5 s** on the ESDF and
occupancy layers so the last field stays on screen; set it by hand (3D panel → Topics → layer →
Decay time) if you built your own layout. Displayed data can then be up to ~2.5 s older than the CAT diagnostics.

`policy_ready=false`, `motion_enabled=false` and `map_alignment_verified=false` are by design.
Guidance arrows (`/g1_cat/guidance_preview`) need a goal on `/g1_cat/goal_preview`; the ESDF does not.

## Commands

```text
perception.sh build          build everything: g1-cat-bridge, g1-cat-rosbridge, repo CAT and repo SLAM images (needs robot internet)
perception.sh build-slam     only the repo SLAM image (Fast-LIO, NDT, Nav2; the slow one)
perception.sh validate       repo's offline CAT tests in the container (no network)
perception.sh start          bridge → CAT research mode (ESDF) → rosbridge
perception.sh start-preview  same, CAT perception-only mode (no fields)
perception.sh viz-start|viz-stop   rosbridge only
perception.sh status|logs|stop
```

Each container is label-checked before stop/remove; other containers are never touched.
Set `ROSBRIDGE_ADDRESS=0.0.0.0` before `start` to connect without a tunnel
(`ws://10.180.68.245:9090`); only do that on a trusted network.

## Rebuilding (only after code changes)

The robot's default route points at a non-existent wired gateway, so it has no internet
until you remove that route (it comes back after reboot):

```bash
sudo ip route del default via 192.168.123.1 dev eth0
```

Then, from the laptop, push the module and rebuild on the robot:

```bash
rsync -a --delete --exclude .build_context modules/g1_cat_bridge/ g1-wifi:workspace/artemvoronov/g1_cat_bridge/
ssh g1-wifi 'cd ~/workspace/artemvoronov && bash g1_cat_bridge/scripts/perception.sh build'
```

Building on the Jetson is fast: the full `build` (bridge, rosbridge, CAT, SLAM) took 17.5 min on 2026-10-21. Cross-building
the ARM64 images on the x86 laptop would need QEMU emulation and is slower.

## Verified on 2026-10-19

- Repo checks on the laptop: 7/7 repository tests, 2/2 launcher tests.
- Robot: `container.sh validate` → 77 tests OK (2 host-only skipped); repo source checks 7/7;
  SLAM image `run_container.sh validate` OK (numpy, scipy, open3d 0.16.0, rclpy, Fast-LIO, NDT, Nav2 packages).
- Live: converter 10.0 Hz × ~20 k points, IMU 200 Hz, 0 rejected; CAT `PREVIEW_OK`,
  ESDF `DISTANCE_ONLY_NO_GOAL` with ~169 k known-free cells; data received on the laptop
  through rosbridge (`/g1_cat/sdf_slice`, `/g1_cat/occupancy_cat`, diagnostics).
- CAT processing ≈ 1.5 s per update and fields ≈ 1.0 s on the Jetson, so displays refresh at ~0.5–1 Hz.
