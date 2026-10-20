# real_robot — the CAT whole-body policy on our G1

Branch `feature/g1-real-robot` (from `feature/cat-wholebody-policy`). Everything needed to run the
sim-trained whole-body policy (`cat/wholebody`, 226 observations → 29 joints) on our Unitree G1,
up to **policy tests on virtual scenes**: the robot walks and localises in the real room, while the
policy only sees boxes of a training scene placed in the map (`esdf_map_virtual`).

What changed in the repository on this branch:

| Path | Change |
|---|---|
| `src/fast_lio` | replaced by **FAST_LIO_G1** (commit `916a59c`, see `UPSTREAM_REVISION`, `manifests/fast_lio_g1.sha256`) |
| `src/g1_sensor_restamp` | new: IMU restamping node (`/utlidar/imu_livox_mid360` → `/g1/imu`) by R. Voronov |
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

Build once (needs robot internet): `real_robot/scripts/g1.sh build all`, then `g1.sh validate`.

1. **Perception check, standing still.** `g1.sh slam odom`, then
   `real_robot/cat_bridge/scripts/perception.sh start-with-slam` for the CAT ESDF in Foxglove.
   Walk the robot with the Unitree remote: the ESDF and the pose must stay consistent.
2. **Map.** Robot on the floor mark, still. `g1.sh slam mapping room1`, walk slowly through the room and
   back to the mark, `g1.sh slam stop` (writes `runtime/slam/maps/room1.{pcd,yaml}`).
   Next sessions: start on the same mark, same heading, `g1.sh slam localize room1`.
3. **Shadow.** `g1.sh wb shadow --scene sidegap_corridor_048 --anchor-at-robot`. Walk the robot with the
   Unitree controller; check `/g1_wb/status` (latency, step time, fields) and the logs. No motor output.
   Foxglove: `cat_bridge/scripts/perception.sh viz-start` (rosbridge), SSH tunnel as in `cat_bridge/README.md`,
   layout `real_robot/foxglove/wholebody_layout.json` (virtual boxes, field slice, pelvis, goal, status).
4. **Virtual scenes, active** — follow the checklist below; first run with `flat_walk_4m`.

### Checklist for the first active run (gantry)

- [ ] `g1.sh validate` passes on the robot; `g1.sh wb shadow` ran ≥ 5 min: step < 15 ms, no overruns, latency < 20 ms
- [ ] LiDAR mount decided (see open items), `--mount` set accordingly
- [ ] Gantry holds the robot, feet ~3 cm above the floor; a second person holds the E-stop
- [ ] Robot in **development (debug) mode** with the Unitree remote, so the vendor controller no longer commands
- [ ] Free floor measured in the map; `--geofence` inside it with ≥ 0.5 m margin; the virtual scene inside the geofence
- [ ] `g1.sh slam localize <map>` (or `odom`) running, pose checked in Foxglove
- [ ] `g1.sh wb active --scene flat_walk_4m --anchor-at-robot --geofence …`, then: `g1.sh op stand` (3 s) →
      lower until the feet carry weight → `g1.sh op balance` → slacken the rope → `g1.sh op go`
- [ ] Any doubt: **L2+B** on the remote or `g1.sh op damp`

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

What it took (each found by this loop, none visible in `cat/wholebody_sim.py`, which updates the pose every step):
* **IMU heading between SLAM updates.** With the 10 Hz held, noisy pose alone the policy clipped the corridor
  walls in 2 of 2 runs and fell once; with heading propagated by the pelvis IMU: 0 contacts.
* **Prefetching the next waypoint's fields.** With 2.5 s solves the robot reached a waypoint before its fields,
  the training stop latch froze it (flat 4 m: 0/3 reached); prefetch + re-arm on new fields: all reached.
* **Fast kinematics.** The training FK re-parsed the MJCF on every call (~6 ms, 3 calls per step) and tripped
  the overrun watchdog; same results, 10x faster.


## Open items before the first active run

1. **ROS/Docker side not run yet**: the images, `run_slam.sh`, the ROS nodes and the bag replay were written
   while Docker was stopped on the laptop. Next: `laptop.sh build`, map + localise on walk1, `laptop.sh sim`.
2. **Jetson timing**: the control step is ~8–11 ms on the laptop; measure in `g1.sh wb shadow` (budget 18 ms).
3. **LiDAR mount**: `robot_xml` (0.419 m, 0° pitch) vs Unitree URDF (0.428 m, 2.9° pitch). Record a bag with
   `/lowstate` to calibrate; walk1 has no joint states.
4. **`/lowcmd` QoS and type on the robot**: check with `ros2 topic info -v /lowcmd` against the vendor subscriber
   before the first active run (publisher is RELIABLE, depth 1).
5. **Development mode**: confirm on our G1 how the vendor controller is released (remote, or MotionSwitcher
   `ReleaseMode`); the executor deliberately never switches modes itself.
6. **License** of `src/g1_sensor_restamp` (`TODO` in its package.xml) before publishing this branch.
