# FAST_LIO (lab fork)

Lidar-inertial odometry for ROS 2 (Foxy or newer), based on [FAST-LIO2](https://github.com/hku-mars/FAST_LIO)
(hku-mars, branch `ROS2`). Configs are provided for the Livox Mid-360, and for the Unitree G1 with its Mid-360
in the head; other lidars with per-point timestamps work through the generic PointCloud2 reader.

Papers: [FAST-LIO2: Fast Direct LiDAR-inertial Odometry](https://arxiv.org/abs/2107.06829) (T-RO 2022),
[FAST-LIO](https://arxiv.org/abs/2010.08196) (RA-L 2021).

## Differences from FAST-LIO2

- **Voxel map** (`map_type: voxel`, default; `ikdtree` is FAST-LIO's map). Each voxel keeps range-weighted
  point statistics (weights 1/r², which observations the average prefers) and, from them, a distribution: the
  spread of its points plus the uncertainty of their mean, `point_sigma`² / n_eff with the effective sample
  size n_eff = (Σw)² / Σw² (how precise it is). Nothing switches at a point count: a voxel is matched on its own
  once its spread is resolved above that uncertainty in two directions (`shape_ratio`, and at least
  `min_n_eff` effective points, the minimum for a covariance), and freezes once its position is known to
  `freeze_sigma`. Where no voxel around a point is resolved, the surface comes from the pooled statistics of
  the 27 or 125 voxels around it (FAST-LIO's neighbourhood plane fit, without stored points), as one surface if
  the region is one (`pooled_thickness`, `pooled_planarity`), else as a weak blob when little else matches
  (`mapping.blob_if_below`). A scan voxel with `scan_patch.min_points` points is one measurement (centroid and
  covariance, plus `point_sigma`² / n and `sigma_floor`), sparser ones keep their points. Among the candidate
  voxels the most likely one is chosen (not the vaguest), and the gate includes the prior's pose uncertainty
  (`mapping.prior_gate`). A planar map distribution (`scan_patch.planar_ratio`) gives one residual along its
  normal, whose variance carries the normal's uncertainty times the in-plane lever arm: its in-plane components
  would only say where the grid cut the plane, and in a corridor they add up to false information along the
  axis. So a sparser scan gives fewer, coarser measurements and a denser one more and finer ones with the same
  rules. See `include/voxel_map.h`, `include/scan_patches.h`.
- **What moved** (`voxel_map.free_space`, on by default): each scan casts rays (a random 15 % of its points,
  `ray_share`, walked on `num_threads` threads) through the voxel map. A ray that passes through a voxel's
  distribution and ends past it (beyond the surface it crossed, for a planar voxel or one on its neighbourhood's
  plane, else beyond the distribution) saw through it; a scan point that fits it confirms it. A voxel seen through
  in `remove_after` (5) scans in a row without a confirmation is removed, frozen or not: what a walking person
  leaves behind. The same rays and points keep evidence of free space in cells, `subdivision` (4) per voxel edge
  (8.75 cm; 0: no cells; `include/free_cells.h`): per scan a cell is hit (a point in it) or missed (a ray crosses it); a
  never-observed cell is free after 7 misses, a long-occupied one after 20, a long-free one is occupied again after
  5 hits, and a hit outweighs 9 misses. A point falling in a free cell is kept out of the map, so a walking person
  mostly never enters it, and stray points in free space (mixed pixels at edges) stay out too. Rays stop marking
  cells free where they enter the band of the surface they end on, and never mark free the cells in a voxel's own
  surface band (grazing views of a floor). The cells only keep new points
  out; what is already in a voxel leaves only with the voxel. A person who never leaves, or whose place stays
  hidden, stays (no free-space method can tell). Cost indoors on the Mac: about 1 ms per scan on one thread,
  0.6 ms on four; the odometry is published before the map update. Scenes with much movement (crowds) would need
  new voxels held for a scan or two before they enter the map; not included.
- **IMU handling.** Every IMU sample is integrated when it arrives; a scan is processed on the state rolled
  back to its end, and the newer samples are integrated again after the update (`src/IMU_Processing.hpp`).
  Live, the IMU, the lidar input and the scan processing each have their own thread.
- **Initialisation.** Gravity and gyro bias are averaged over `mapping.imu_init_time` (stand still), and the
  world frame is gravity-aligned (z up) from the start. The initial covariances of the biases and of gravity
  are parameters (`mapping.init_cov_*`).
- **Lidar-IMU rotation.** The Mid-360's internal lidar-IMU rotation is off by up to about 0.5 deg per unit.
  Walking straight, the IMU-integrated velocity is then rotated against the lidar's motion, the per-scan
  discrepancy leaks into pitch, and a long corridor bends. Online extrinsic estimation drifts on a walking
  robot; calibrate `mapping.extrinsic_R` once instead. Replay a long straight out-and-back walk (`--replay`) over a
  grid of pitch and roll angles (R = Rx(roll) Ry(pitch), lidar to IMU), and keep the angle whose trajectory has the
  smallest vertical range and bending: the bend grows linearly with the angle error.
- **Accelerometer saturation** (`mapping.acc_range`): readings at the sensor's range get a large noise.
- **Outputs** at IMU rate (odometry, TF), and for navigation stacks a level "sensor" frame and the registered
  scan, with configurable topics. The trajectory is `/trajectory` (`/path` is what navigation stacks call the
  planned path).
- Generic PointCloud2 input (any per-point time field and unit), random point thinning (`point_keep_ratio`),
  timing log, offline replay and an interactive debug session.

## Build

Needs PCL, Eigen and the `livox_ros_driver2` messages (a message-only copy works); zlib for the offline replay.

```bash
colcon build --packages-up-to fast_lio --cmake-args -DCMAKE_BUILD_TYPE=Release
```

`-DIKD_SYNC_REBUILD=ON` makes `map_type ikdtree` deterministic (synchronous tree rebuilds), for regression checks.

## Run

```bash
ros2 launch fast_lio mapping.launch.py config:=$(ros2 pkg prefix fast_lio)/share/fast_lio/config/mid360.yaml
ros2 launch fast_lio mapping.launch.py outdoor:=true     # loads config/outdoor.yaml after the config
```

The default config is `g1_mid360.yaml`. Keep the sensor still for `mapping.imu_init_time` (3 s) after the start.

| Config | Use |
|---|---|
| `config/mid360.yaml` | Livox Mid-360 with `livox_ros_driver2` (CustomMsg). **The reference for all parameters**: start a new robot from it |
| `config/g1_mid360.yaml` | Unitree G1: the vendor's Mid-360 stream (PointCloud2) with `imu_restamp`, navigation outputs |
| `config/dataset_mid360.yaml` | Offline evaluation on recorded datasets (replay topics) |
| `config/outdoor.yaml` | Outdoor overrides (larger voxels), loaded after one of the above |

For a new robot: set the topics and `preprocess` (message type, time field and unit), `blind` (the robot's
body), the extrinsic (lidar pose in the IMU frame), `acc_range` (accelerometer range in the IMU's units), the
IMU noise (a walking robot needs much more `acc_cov` than a cart) and the outputs. If the IMU driver stamps each
sample with host time on receipt (intervals that stall and then catch up in a burst, repeated samples), enable
`imu_restamp`: it puts the stamps back on the sensor's own sample grid before anything else sees them
(`include/imu_restamper.h`; the counters are logged every 30 s of data).

## Outputs

Frames (`frames.*`, names configurable):

```
map -> init -> odom -> imu -> lidar
                          \-> lidar_level
```

| Frame / TF | Content |
|---|---|
| `init` | the filter's frame: gravity-aligned (z up), origin at the IMU at the start |
| `map -> init` | static. SLAM: identity (left out with `frames.map: ""`). Localization: the pose of `init` on the primary map, from the first localization |
| `init -> odom` | the jumps of the estimate, identity until the first: a relocalization, or an update that moves the pose by more than `publish.jump_distance` / `jump_angle_deg` (0.25 m / 3°: above every normal correction, at most 14 cm / 2.3° over 22 sequences, 1.2 cm / 0.8° on the G1). Sent with each odometry message |
| `odom -> imu` | the filter's IMU pose, continuous; sent with each odometry message |
| `imu -> lidar` | the configured extrinsic, static |
| `imu -> lidar_level` | static, from the end of the initialisation: the lidar origin with the axes of `init` at that moment (level while the robot stood, x the IMU's heading), whatever the mounting: a body frame for navigation and control |

| Topic | Rate | Content |
|---|---|---|
| `odometry_topic` (`/Odometry`) | scan, or every N-th IMU sample | pose and twist of `odometry_frame` (`imu`, `lidar` or `lidar_level`) in `odometry_parent` (`odom`, or `map`), with covariances; always the latest estimate (`odometry_imu_every`: 0 the updated pose at the scan end, N the propagated state at every N-th IMU sample) |
| `registered_cloud_topic` | scan | the full deskewed scan in `odometry_parent` (`publish_registered_cloud`) |
| `/cloud_registered`, `/cloud_registered_body` | scan | the scan in `odometry_parent` / `imu` (`scan_publish_en`, `dense_publish_en`, `scan_bodyframe_pub_en`) |
| `/trajectory` | 1/2 scans | the last 5000 scan-end IMU poses in `odometry_parent` (`path_en`) |
| `map_topic` (`/Laser_map`) | every `map_period` (1 s) of scan time | the map in `odometry_parent` (`publish_map`), only while someone subscribes: each voxel's distribution as points `map_point_step` (0.1 m) apart, its principal box of ±√3σ cut to the voxel, one layer on the plane of a planar voxel (0: the voxel means); in localization the live layer, the saved map is `/prior_map` (same points). Colored by the plane normal, muted (`rgb`: walls facing x red, facing y green, floors and ceilings blue-gray, slopes in between; gray: not planar; Foxglove color mode RGB), intensity \|n_z\| (0.5: not planar). ikd-Tree: its points |
| map file | at the stop | with `map_save: true`, when the node stops (SIGINT; it leaves ROS first, so a new stack can start meanwhile), writes the map to `map_file_path`, the extension set by `map_type`: `.vmap` (the voxel map itself; `scripts/vmap_to_pcd.py` converts it for point cloud viewers) or `.pcd` (the ikd-Tree's points). A SLAM map is saved level: rotated about its origin so that the final gravity estimate points along -z (`init` is level only as the accelerometer found it at the start, which its bias tilts; the gravity state corrects that once the robot has moved), the angle in the log; the run's odometry stays in `init`. In localization mode the live layer, never over an existing file: `<name>_local_map_<n>.vmap` |

Timestamps are the scan end or the IMU sample. Pose covariances are in ROS order (position, then rotation about
the parent frame's axes); the twist is in the child frame. The `lidar` frames use the configured extrinsic.

## Localization on a saved map

1. Map: run SLAM with `map_save: true` and `map_file_path: "<room>"`: the voxel map is written when the node stops
   (`<room>.vmap`); or `fastlio_mapping --replay <dataset> <odom.tum> --map <room>` offline. The map is saved level
   (the final gravity estimate, see `map_save`), as the relocalization needs (it searches x, y, z and yaw only), so
   stop it after the robot has walked, not while it still stands at the start. A wrong file stops the node with the
   reason: not found, a PCD (ikd-Tree maps: localization not implemented yet), another version, damaged.
2. Localize: start with `-p localization.map_file:=<room>.vmap` (`map_type: voxel`). The primary map is
   published latched on `/prior_map` (frame `map`), the state on `/localization_status`.
3. The start: without `localization.initial_pose`, the whole map is searched once, as soon as the first
   `submap_scans` scans are in (about 0.1 s on a room's map). If that fails (the robot is not on the map, an
   ambiguous place), the status asks for a pose: `/initialpose` in the map frame (Foxglove: 3D panel, display
   frame `map`, pose estimate tool), within 2 m and 45° (`search_xy`, `search_yaw_deg`) of the truth. With
   `initial_pose: [x, y, yaw_deg]` only the search near it runs. Another pose relocalizes at any time.

Until the first localization the filter runs as SLAM in `init`. A relocalization (`include/relocalization.h`)
takes the last `submap_scans` scans in the level frame of the newest, so only x, y, z and yaw are unknown:
- z: the map's floor plus the robot's height above its own floor (one floor per map); `search_z` checks it for
  sensors that see the floor only far away (high and upright);
- x, y, yaw: branch and bound over the map's voxel grid for every yaw step (one voxel at `max_range`, 25 m), over
  the whole map or within `search_xy` / `search_yaw_deg` of a guess. A pose scores the voxel values of the points
  in `score_band` (0.2-2.5 m above the floor; 2 occupied, 1 next to an occupied voxel; floor and ceiling points fit
  anywhere); blocks of translations are bounded by the maximum of the values over the block;
- the best distinct poses are refined on the map's distributions (Gauss-Newton on x, y, z, yaw), and the best is
  accepted only if at least `accept_matched` of the points match it and no distinct pose matches
  `max_ambiguity` of that: a symmetric place (a long corridor) is reported as ambiguous, not guessed.

About 0.1 s over a room's or a corridor's map, 20-40 ms near a guess (Mac). Live it runs beside the filter; the
offline replay waits for it, so a replay is repeatable and writes the poses in `map`. Afterwards the scans are
matched against the primary map (never changed) and a live layer that takes only what the map lacks (unknown
space, new objects), stays within `live_radius` of the robot and is cleared at each relocalization
(`src/localization.hpp`). The first localization sets `map -> init`; a later one moves the state, and the move goes
into `init -> odom`.

## Timing log

With `runtime_pos_log_enable`, one line per scan is printed, and at exit `runtime_pos_log_path` (default
`fast_lio_time_log.csv` in the working directory) gets one row per scan: `time_stamp` [s], `total time` [s]
(deskew to map update), `preprocess time` [s], `scan point size`, `down points`, `map size` (points or voxels),
`matched`, `update calls`, `residual mean` [m]; then the state after the update: `grav_tilt_deg` (gravity state
against -z), `roll_deg`, `pitch_deg` (IMU in the world frame), `ba_*`, `bg_*` (biases), `ext_r*` (lidar-to-IMU
rotation estimate, rotation vector [deg]); and the lidar information of the last update iteration: `lam_t0..2`,
`lam_r0..2` (eigenvalues of H'H over the translation, world frame, and the rotation, body frame, ascending),
`weak_t*` (the weakest translation direction), and per down
point of the last iteration `no_match` (no resolved surface around it), `gate_fail`, `pooled` (matched to a pooled
neighbourhood's surface) and `blob`; `free_space_time` [s] and `removed` (voxels seen through, `voxel_map.free_space`).
`lam_t0 / lam_t2` is a scale-free degeneracy measure: 0.3-0.5 in rooms, 0.04-0.15 in a featureless corridor.

## Offline replay and debug session

```bash
fastlio_mapping --replay <dataset.g1replay> <odom.tum> [--map <name>] [--debug-port <port>] \
    [--dump-scans <file>] --ros-args --params-file config/dataset_mid360.yaml
```

feeds a recorded dataset straight into the node (no DDS, as fast as it computes, same result as a live run
that drops nothing) and writes the odometry as TUM (in `init`; in localization mode in `map`, from the first
localization on). Datasets are made from rosbags by `scripts/make_replay.py`. `--debug-port` stops before a
chosen scan's update for a viewer (`scripts/lio_viewer`; `src/debug_session.hpp` describes the protocol): the
prior, the scan, the map and every update iteration per point can be inspected, and the update rerun from a
perturbed prior or with other parameters. `--dump-scans` writes every scan as it went into the voxel map (the
deskewed points in the lidar frame, the lidar pose, which points were inserted, the voxels removed before it; format
in `src/laserMapping.cpp`, `scan_dump`), for analyses outside the node, e.g. rebuilding the map offline.

## Scripts

| Script | Use |
|---|---|
| `scripts/make_replay.py` | a dataset bag (ROS 1 or 2; presets for TIERS, HR²-KILO, M3DGR, JEPLO, the G1) into a `.g1replay` for `--replay` (numpy, rosbags) |
| `scripts/vmap_to_pcd.py` | a `.vmap` into a PCD for point cloud viewers: voxel means with counts, normals and planes (numpy) |
| `scripts/vmap_level.py` | levels a saved `.vmap` by its floor (a map saved before the leveling, or a sloped one): the whole map rotated as `map_save` does, `--tilt-only` reports the tilt (numpy) |
| `scripts/lio_viewer/` | the browser viewer of the debug session: the maps, the scan, the update per point (Python stdlib; its README) |

## License

GPL-2.0, see `LICENSE`.
