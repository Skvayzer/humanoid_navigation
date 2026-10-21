# real_robot — the CAT whole-body policy on our G1

Branch `feature/g1-real-robot` (from `feature/cat-wholebody-policy`). Everything needed to run the
sim-trained whole-body policy (`cat/wholebody`, 226 observations → 29 joints) on our Unitree G1,
up to **policy tests on virtual scenes**: the robot walks and localises in the real room, while the
policy only sees boxes of a training scene placed in the map (`esdf_map_virtual`).

What changed in the repository on this branch:

| Path | Change |
|---|---|
| `src/fast_lio` | replaced by **FAST_LIO_G1** (commit `f194a59`: voxel map, G1 IMU noise; see `UPSTREAM_REVISION`, `manifests/fast_lio_g1.sha256`) |
| `src/g1_sensor_restamp` | new: IMU restamping node (`/utlidar/imu_livox_mid360` → `/g1/imu`) by R. Voronov, MIT |
| `real_robot/` | new: everything below |
| `tests/test_repository.py` | checks `src/fast_lio` against the FAST_LIO_G1 manifest instead of the old snapshot |

The old snapshot pipeline (`docker/Dockerfile`, `scripts/run_stack.sh`, `config/mid360_g1.yaml`) is left as
captured but is **superseded here**: it waits for the old Fast-LIO's gravity-aligned `world` frame, which
FAST_LIO_G1 does not publish.

## Pipeline

```text
vendor main board ── /utlidar/cloud_livox_mid360 (10 Hz) ──────────────┐
                 └── /utlidar/imu_livox_mid360 ── g1_sensor_restamp ── /g1/imu (200 Hz grid)
                                                                        │
                       g1-rr-slam:  FAST_LIO_G1 ── camera_init→body (/g1_slam/odometry, 10 Hz)
                                    + floor_anchor (odom mode)  or  NDT in a saved map (localize mode)
                                    ── TF map→camera_init (map: gravity up, floor z = 0)
                                                                        │
/lowstate (500 Hz) ─────────── g1-rr-wholebody:  pelvis pose = SLAM body · mount⁻¹ · waist FK⁻¹,
                                                 heading/position from SLAM, roll/pitch from the pelvis IMU
                                                 virtual scene → fields (separate process, prefetch)
                                                 observation → ONNX policy → targets + kp/kd + gravity ff
                                                 watchdogs → /lowcmd (only with --active)
```

## Layout

| Path | What |
|---|---|
| `wholebody/g1_wholebody/` | executor: `lowlevel` (LowCmd+CRC, raw LowState, remote), `frames` (pelvis chain), `kinematics` (fast FK), `fields` (virtual scenes, worker), `runner` (modes, watchdogs, control step), `ros_executor` (ROS node) |
| `wholebody/g1_wholebody/` (dev) | `fake_robot` (MuJoCo G1 behind LowCmd/LowState), `sim_loop` (closed loop on simulated time), `sim_matrix`, `ros_fake_robot` |
| `wholebody/tests/` | 30 tests: CRC vs the Unitree SDK library, CDR layout, chain vs MuJoCo, FK identical to training, fields vs the sim, safety behaviour |
| `slam/` | `run_slam.sh` (odom / mapping / localize), `floor_anchor.py`, `map_builder.py`, `record_traj.py`, tests |
| `config/` | `fastlio_g1.yaml` (colleague's config + `[RR]` lines), `ndt_g1.yaml`, `cyclonedds_laptop.xml` |
| `ros/unitree_hg/` | G1 messages incl. LowCmd/MotorCmd (executor image only; the CAT image keeps its telemetry-only copy) |
| `docker/` | `slam.Dockerfile`, `wholebody.Dockerfile` (targets `runtime`, `dev`) |
| `cat_bridge/` | CAT perception preview (ESDF in Foxglove): LiDAR converter, stationary pose, rosbridge |
| `scripts/g1.sh` | robot entry point · `scripts/laptop.sh` laptop entry point |

## Safety design of the executor

Modes: `IDLE` (publishes nothing) · `SHADOW` (policy on live data, never publishes) · `STAND_UP` (3 s ramp to
the nominal pose, kp 20 → 100 %) · `STAND` (PD hold, **gantry only**, no balance) · `POLICY` · `DAMP`
(kp = 0, kd = 2 × contract kd, latched until `reset`). Active modes publish every step, DAMP included: the motor
drivers keep executing the last command they received.

| Watchdog → DAMP | Threshold (`runner.Limits`) |
|---|---|
| LowState stale / non-finite | > 50 ms |
| SLAM pose stale (POLICY) | > 0.35 s |
| SLAM pose jump (POLICY) | > 0.15 m + 1.5 m/s·dt or 10° + 300 °/s·dt between updates |
| fields stale (POLICY) | > 8 s |
| control step overrun | > 40 ms once, or > 18 ms for 5 steps in a row |
| pelvis tilt (STAND, POLICY) | > 45° (pelvis IMU) |
| pelvis height (POLICY) | < 0.45 m |
| joints beyond soft limits (STAND, POLICY) | > 0.2 rad |
| outside the geofence (POLICY) | always; POLICY is refused without a geofence |
| motor temperature | > 90 °C |
| policy output | non-finite / wrong size |
| run time (POLICY) | > 180 s |

Remote (Unitree controller, read from `LowState.wireless_remote`): **L2+B = DAMP at any time** ·
L1+A held 0.5 s = stand · R1+A held 0.5 s = go · R1+B held 0.5 s = stop (stand in place).
Without `--active` the `/lowcmd` publisher is never created. `g1.sh wb active` asks to type `ACTIVE` and
requires `--geofence`. Stopping the node while active sends 0.5 s of damping first.

Not covered by software: a fall in DAMP (that is why the gantry), motors that ignore commands, the vendor
controller still running (the operator puts the robot in development mode, see the checklist).

## Laptop

```bash
cd g1_traversal_policy/modules/humanoid_navigation
PYTHONPATH=real_robot/wholebody:cat ../../.venv/bin/python -m unittest discover -s real_robot/wholebody/tests
(cd real_robot/slam && PYTHONPATH=. ../../../../.venv/bin/python -m unittest discover -s tests)
PYTHONPATH=real_robot/wholebody:cat ../../.venv/bin/python -m g1_wholebody.sim_loop --procedure gantry \
    --scene sidegap_corridor_048 --video out.mp4 --assets ../../data/third_party/unitree_ros/robots/g1_description/meshes
PYTHONPATH=real_robot/wholebody:cat ../../.venv/bin/python -m g1_wholebody.sim_matrix --out ../../results/closed_loop
# with Docker (ROS 2 Foxy amd64):
real_robot/scripts/laptop.sh build && real_robot/scripts/laptop.sh validate
real_robot/scripts/laptop.sh map ../../data/robot_bags/20261021_013309_walk1 walk1     # map from the bag
real_robot/scripts/laptop.sh localize ../../data/robot_bags/20261021_013309_walk1 walk1
real_robot/scripts/laptop.sh sim                                                       # ROS loop vs simulated G1
```

## Robot (from the checkout in `~/workspace/artemvoronov`)

**Images, preferred: build on the laptop, load on the robot** (no robot internet, no compile on the Jetson):

```bash
# laptop, once: sudo apt install qemu-user-static binfmt-support   (arm64 emulation for Docker)
real_robot/scripts/laptop.sh build-robot      # arm64 images under QEMU, emulated tests, archive
rsync -P ../../results/laptop/images/g1_rr_images_arm64.tar.gz g1:workspace/artemvoronov/cache/
# robot:
real_robot/scripts/g1.sh load ~/workspace/artemvoronov/cache/g1_rr_images_arm64.tar.gz
real_robot/scripts/g1.sh validate             # native tests incl. the timing checks skipped under QEMU
```

SLAM image: 1.6 GB (was 3.7 GB): the runtime stage installs only the 12 PCL libraries the binaries link; `pcl_ros`/`pcl_conversions` are not linked but pulled all of `libpcl-dev` (PCL/VTK/Qt headers). rosbag2 is in the laptop-only `dev` stage. `build-robot` builds both images in parallel.
Full emulated build from scratch: 97 min (`lidar_localization_ros2` alone 43 min); arm64 SLAM image 1.45 GB, executor 0.95 GB, archive for the robot 0.6 GB. A rebuild without changes takes seconds (all layers cached). The SLAM builder has one layer per
package, ordered from stable/expensive to frequently changed: `livox_ros_driver2` → `ndt_omp_ros2` →
`lidar_localization_ros2` (~40 min under QEMU) → `g1_sensor_restamp` → `fast_lio` (the one being updated). A change rebuilds only that
package and the ones after it, and ccache (BuildKit cache mount) recompiles only the files that changed.
Changes in `real_robot/` touch only light copy layers. On the robot's Docker 24 without buildx, `g1.sh build`
strips the cache mounts (per-package layers still apply, ccache does not persist).
Under QEMU the code runs ~10–17× slower, so `build-robot` runs the tests with `G1_TIMING_TESTS=0`; the
step-time watchdog itself is covered by a deterministic test.

Alternative (needs robot internet): `real_robot/scripts/g1.sh build all`, then `g1.sh validate`.

1. **Perception check, standing still.** `g1.sh slam odom`, then
   `real_robot/cat_bridge/scripts/perception.sh start-with-slam` for the CAT ESDF in Foxglove.
   Walk the robot with the Unitree remote: the ESDF and the pose must stay consistent.
2. **Map.** Robot on the floor mark, still. `g1.sh slam mapping room1`, walk slowly through the room and
   back to the mark, `g1.sh slam stop` (writes `runtime/slam/maps/room1.{pcd,yaml}`).
   Next sessions: start on the same mark, same heading, `g1.sh slam localize room1`.
   Alongside, `g1.sh wb shadow --scene flat_walk_4m --anchor-at-robot` and walk the robot along the scene once:
   its log gives the Jetson step time while fields are solved and the data for an offline mount check.
3. **Shadow.** `g1.sh wb shadow --scene sidegap_corridor_048 --anchor-at-robot`. Walk the robot with the
   Unitree controller; check `/g1_wb/status` (latency, step time, fields) and the logs. No motor output.
   Foxglove: `cat_bridge/scripts/perception.sh viz-start` (rosbridge), SSH tunnel as in `cat_bridge/README.md`,
   layout `real_robot/foxglove/wholebody_layout.json` — the same view in shadow and active runs:
   * 3D, following the robot (TF `map → g1_wb_pelvis`): the virtual boxes (`esdf_map_virtual`, orange), the ESDF
     slice at 0.75 m (`/g1_wb/sdf_slice`, turbo, −0.1…1.2 m), the 13 body sites the policy samples
     (`/g1_wb/sites`, red = touching a virtual box … green ≥ 0.3 m), waypoints and path (green), geofence (red),
     pelvis and goal arrows. The live scan `/g1_slam/cloud_registered` is off by default (heavy over rosbridge).
   * Indicators: executor mode, and **rope** (`KEEP ROPE: stiff stand / policy waiting / settling`, then
     `standing: rope may slacken`, `walking`, `DAMP`) — derived from `/g1_wb/status` by the layout's user script.
   * Plots: step and LowState→command time (budget 18 ms), fields age / solve time / pelvis-to-field-goal distance;
     the status as fields; `/rosout` of the executor (mode changes, DAMP reasons).
4. **Virtual scenes, active** — follow the checklist below; first run with `flat_walk_4m`.

### Checklist for the first active run (gantry)

- [x] `g1.sh validate` passes on the robot; `g1.sh wb shadow` ran ≥ 5 min: step < 15 ms, no overruns, latency < 20 ms
      (2026-10-21, 15 min: p99 9.7 ms, max 19.1 ms, no watchdog trip — see open items)
- [x] LiDAR mount decided: `unitree_urdf` (stock G1), the default
- [ ] Gantry holds the robot, feet ~3 cm above the floor; a second person holds the E-stop
- [ ] Robot in **development (debug) mode** with the Unitree remote, so the vendor controller no longer commands
- [ ] Free floor measured in the map; `--geofence` inside it with ≥ 0.5 m margin; the virtual scene inside the geofence
- [ ] `g1.sh slam localize <map>` (or `odom`) running, pose checked in Foxglove
- [ ] `g1.sh wb active --scene flat_walk_4m --anchor-at-robot --geofence …`, then: `g1.sh op stand` (3 s) →
      lower until the feet carry weight → `g1.sh op balance` → **wait for the rope indicator
      `standing: rope may slacken`** (status `policy_wait: null`, `latch: 0`: the policy holds the robot and has come
      to rest; it takes ~1–3 s of steps and up to ~0.25 m first) → slacken the rope → `g1.sh op go`
- [ ] Any doubt: **L2+B** on the remote or `g1.sh op damp`. `stop` (R1+B) is not immediate: the robot walks on
      until the fields of the hold point are solved (up to ~5 s on the Jetson)

## Verified so far (laptop, 2026-10-21)

| Check | Result |
|---|---|
| Repository tests (incl. FAST_LIO_G1 manifest) | 8/8 |
| Branch's `test_wholebody` | 8/8 |
| Executor package tests | 30/30 |
| SLAM tool tests | 5/5 |
| `g1_sensor_restamp` offline eval on walk1 vs the live `/g1/imu` in the bag | 99.97 % of stamps within 0.1 ms |
| Floor anchor on walk1's first 4 s | IMU 1.274 m above the floor, floor RMS 0.9 cm, slope 0.44° |
| LowCmd CRC vs the SDK's `crc_amd64.so` | identical on 50 random commands |
| Pelvis chain vs MuJoCo FK | < 1e-9 m |
| Fast FK vs training FK | identical (1e-12), 1.7 ms vs 5.9 ms per call |

### Closed loop: the robot executor against a simulated G1 (`g1_wholebody.sim_matrix`)

Each run does the full gantry procedure (hanging, motors off → `stand` → lower → `balance` → slacken → `go`)
under robot-like conditions: SLAM pose at **10 Hz, held in between**, 60 ms latency, 1 cm / 0.01 rad noise per
update; fields delivered 2.5 s after the request (expected Jetson solve time); commands reach the motors 10 ms
late; motors 10 % weaker (kp × 0.9); virtual scene rotated 0.4 rad in the map. Timing watchdog active.

| Set / policy | flat 4 m | gap 0.47 | corridor 0.48 | gap 0.44 |
|---|---|---|---|---|
| robot-like / `robust_delay_v1_u150` | 3/3 | 3/3 | 3/3 | 3/3 |
| robot-like / `u596` | 3/3 | 3/3 | 3/3 (1 run: elbow probe 2 steps in a wall) | 3/3 |
| true mount = URDF, executor assumes `robot_xml` | 2/2 | 2/2 | 2/2 (1 run: elbow probe 9 steps in a wall) | 2/2 |
| full 20 ms command latency | 2/2 | 2/2 | 2/2 | 2/2 |

40/40 reached the goal, 0 falls, 0 watchdog trips. Longest control step 15.5 ms with 6 simulations sharing the
laptop; alone: POLICY step 2.9 ms median, 3.9 ms max (was 23.5 ms before `FastKinematics`).
**This table was too optimistic** (seeds 0–2 only): 8 seeds per cell later showed two executor bugs, both in
both mounts (flat 4 m 6–8/8, corridor 6/8 reached), fixed on 2026-10-21 (last two items below). After the fix,
smoke check with the URDF mount: flat 4 m and corridor 0.48, seeds 0–1, 4/4 reached, 0 falls, 0 contacts,
0.14–0.16 m drift during `balance` (was up to 1.1 m).

What it took (each found by this loop, none visible in `cat/wholebody_sim.py`, which updates the pose every step):
* **IMU heading between SLAM updates.** With the 10 Hz held, noisy pose alone the policy clipped the corridor
  walls in 2 of 2 runs and fell once; with heading propagated by the pelvis IMU: 0 contacts.
* **Prefetching the next waypoint's fields.** With 2.5 s solves the robot reached a waypoint before its fields,
  the training stop latch froze it (flat 4 m: 0/3 reached); prefetch + re-arm on new fields: all reached.
* **Fast kinematics.** The training FK re-parsed the MJCF on every call (~6 ms, 3 calls per step) and tripped
  the overrun watchdog; same results, 10x faster.
* **New goal while stopping.** The worker solves one request at a time. When a waypoint switch came before its
  fields, the policy stopped on the old fields (goal = the waypoint just reached); fields for the new goal arriving
  during the 1 s stop phase were ignored (re-arm only after a full stop), and from then on the goal never changed:
  standing for good, 15–25 % of runs. Now a new goal during the stop phase walks on (gait phase kept).
* **`balance` walked.** The hold request at `balance` was refused while the worker was still busy, and the policy
  started on the first waypoint's fields: ~1 m of walking with the rope still on. Now `stand` already solves
  "stay here" (the hold point follows the robot on the gantry), and the policy starts only on hold fields;
  otherwise it keeps the stand pose and the status shows `policy_wait`. The sim reports `hold_drift_m`.


### ROS 2 side on the laptop (Docker, Foxy amd64, 2026-10-21)

| Check | Result |
|---|---|
| Executor image (`laptop.sh validate`, Python 3.8, numpy 1.24, ONNX Runtime 1.16.3) | `test_wholebody` 8/8 incl. ONNX action parity, executor tests 30/30 (SDK-library test skipped) |
| Raw LowState decoder vs `rclpy.serialization` | identical (q, dq, tau, IMU, temperatures, remote, tick) |
| walk1 replay → restamp → FAST_LIO_G1 → floor anchor → map builder | map 226 k points, 33.3 m path, floor RMS 1.7 cm |
| End-position error (ICP of first/last scans, as in the report) | **7.2 / 10.7 / 8.3 cm** in three real-time replays (4.7 / 7.3 / 6.3 cm horizontal, 0.6–1.1°); report: 7.8 cm (4.7 cm horizontal). Real-time playback + 3-thread matching is not deterministic |
| NDT localisation in that map, same bag | vs the mapping trajectory: 1.8 cm mean, 4.2 cm p95, 4.9 cm max |
| **FAST_LIO_G1 `f194a59`** (voxel map 0.35 m, `acc_cov` 0.3, keep ratio 0.5) on walk1 | end error **0.9 / 0.6 cm**, 0.29° (two replays; `916a59c`: 7.2–10.7 cm); map 240 k points, 32.1 m, floor RMS 1.6 cm; NDT vs mapping trajectory 0.9 cm mean, 1.5 cm max |
| `ros_executor` vs `ros_fake_robot` over real topics, gantry procedure, corridor 0.48 m | reached the goal; 50 Hz loop: period 20.0 ms median / 20.7 ms max, step ≤ 8.8 ms, LowState age ≤ 12.8 ms, 0 overruns |
| Remote L2+B over ROS | DAMP at once; damping commands (kp 0, kd ≥ 2, CRC) keep coming at 50 Hz |
| Executor CPU with LowState at 500 Hz (as on the robot) | main process 17 % of a core idle, 30 % in POLICY (was ~100 %); field worker 100 % while solving |
| LowCmd CRC | SDK library `crc_<arch>.so` (checked against the Python reference at load): pack + CRC 0.07 ms (was 0.4 ms) |

CPU: `/lowstate` is no longer an rclpy callback (each costs ~0.3 ms of Python on the laptop, 500 per second on the
robot); the loop takes the newest sample itself from a KEEP_LAST-1 subscription on a node the executor never spins,
and otherwise waits to its deadline in one call (Foxy releases the GIL while waiting). Note: Foxy's raw take returns
`(None, info)` when nothing is waiting. `latency_ms` in the status is now measured from that take (the sample itself
is at most 2 ms older at 500 Hz).

Bugs this run found and fixed (all would have hit the robot): Python TF in a namespaced Foxy node uses
relative `tf`/`tf_static` (floor anchor/map builder published into `/g1_slam/tf_static`); a Foxy logger call
site may not change severity (crashed the node at the first DAMP — it still left the motors damped); assigning
`bytes` to a `uint8[]` field costs 40 ms in Foxy (field slice stalled the loop); the in-loop log compression
(~60 ms every 10 s) and a schedule-reset bug stalled the loop until the LowState watchdog fired. Also: rosbag2
(Foxy) cannot open a read-only bag (`laptop.sh` replays a working copy).

## Open items before the first active run

1. ~~**Robot images**~~: done 2026-10-21 — `g1.sh load` of the laptop-built arm64 archive, `g1.sh validate` on
   the Jetson: `test_wholebody` 8/8, executor tests 32/32 (incl. FK timing and the SDK CRC library).
2. ~~**Jetson timing**~~: `g1.sh wb shadow` for 15 min (robot standing, SLAM odom): loop 50.0 Hz, period 20.0 ms
   median / 21.5 ms p99 / 26.4 ms max; step 5.0 ms median / 9.7 ms p99 / 19.1 ms max (4 single steps > 18 ms, never
   consecutive; the watchdog needs 5 in a row or one > 40 ms). Not measured yet: the step while the field worker
   solves (robot walking through waypoints) — run shadow during the mapping walk.
3. ~~**LiDAR mount**~~: our G1 is a stock robot, so `unitree_urdf` (Unitree's `g1_29dof_rev_1_0.urdf`, `mid360_joint`:
   0.428 m, 2.93° pitch) is the default everywhere; `robot_xml` (0.419 m, 0°, the training model's site) remains
   selectable. If it is ever in doubt: the executor log now has the SLAM pose (`slam`, `slam_stamp`) next to the
   joints and the pelvis IMU, so a shadow run during mapping is enough to check the mount offline.
4. ~~**`/lowcmd` QoS and type**~~: on the robot `unitree_hg/msg/LowCmd`, 2 vendor publishers (RELIABLE), 1 subscriber.
5. **Development mode**: confirm on our G1 how the vendor controller is released (remote, or MotionSwitcher
   `ReleaseMode`); the executor deliberately never switches modes itself. Check: after the switch `/lowcmd`
   has no vendor publishers left. On the gantry only (the robot goes limp).
