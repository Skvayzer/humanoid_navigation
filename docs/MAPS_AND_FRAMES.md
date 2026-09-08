# Maps, frames and robot-specific calibration

The deployed TF chain is `map -> world -> camera_init -> body`. NDT supplies
`map -> world`; Fast-LIO supplies gravity alignment and odometry. A separate
planar `map -> g1_nav_base` transform and `/g1_nav/odom` adapt the full 3D body
pose to Nav2. Do not add an identity map/world transform to silence a TF error.

The map must be floor-aligned once, and the NDT startup pose must use the **same
rigid transform**. `/g1_viz/cloud_registered_floor` is a diagnostic RANSAC-aligned
cloud in `camera_init_floor`; it does not change the Fast-LIO estimator or the
saved localization map.

## Critical G1 extrinsic correction

In this robot's vendor Livox driver, cloud coordinates have a 180-degree X roll
while the built-in IMU remains in native sensor axes. Consequently the deployed
Fast-LIO `extrinsic_R = diag(1, -1, -1)` maps that published cloud into the IMU
frame. Translation is `[-0.011, -0.02329, 0.04412]`. These values are preserved in
`config/mid360_g1.yaml`. They are **not universal MID360 defaults**. Applying them
to an unrotated driver's cloud would introduce the error they correct here.

Before navigation, move the supported robot forward manually with the gateway
disarmed, and verify `/g1_nav/odom` forward agrees with positive Unitree `vx`.
The planar bridge derives heading from the projected body X axis, not a naive
yaw extracted from an upside-down/tilted mounting frame.

## New maps

`slam-start-map` collects the ikd-tree map with point_filter_num=1,
filter_size_surf=0.15 m, filter_size_map=0.10 m. Dense visualization publishing is
enabled. Monitor load and estimator health; finer maps cost CPU/memory.

`scripts/align_pcd_floor.py` is the deployed lab-specific binary-PCD converter.
Its plane convention is `z = a*x + b*y + c`, with **negative raw Z as up**. It
prints the rotation, translation and XYZW quaternion to use consistently for the
map/initial pose. Fit the actual floor and check this convention before use;
do not feed it arbitrary coefficients or already-aligned maps. It expects the
Fast-LIO float32 PCD layout, including normals.

`scripts/make_nav2_map.py` projects an aligned binary PCD into a 0.05 m occupancy
grid using floor/free-space support and an obstacle height slice. Install NumPy,
SciPy and Pillow in a dedicated desktop environment for these offline utilities.
Example (choose a NEW output directory to avoid overwriting edited maps):

```bash
python3 scripts/make_nav2_map.py aligned_map.pcd new_nav_map/
```

This retained script names its outputs `teaching_lab_nav.pgm/.yaml` even for a
new input; set `G1_NAV_MAP_YAML` to their actual location after copying. Manually
edited occupancy grids must be backed up separately: regenerating the grid does
not preserve edits. Do not draw free space over real hazards.

NDT is local registration, not automatic global localization. Being within a
fixed radius of the original start does not guarantee convergence; heading,
overlap and scene geometry matter. For another start, set a suitable 6-DOF initial
pose in the map frame or use the localizer's `/initialpose` input through a
trusted ROS tool. The visualization bridge deliberately does not allow publishing
`/initialpose`. Verify scan/map overlap and NDT diagnostics before arming.

## Current obstacle behavior

- Global costmap: saved occupancy grid + inflation, robot radius 0.35 m,
  inflation radius 0.42 m.
- Local costmap: rolling 5x5 m, 0.05 m resolution, 5 Hz, live PointCloud2 from
  `/g1_nav/lidar_costmap`, obstacle height 0.15–1.80 m, marking to 2 m and clearing
  to 2.5 m, inflation radius 0.55 m.
- Controller: Foxy **Regulated Pure Pursuit**, not MPPI. It follows the global
  path with collision checking/regulation; it does not have MPPI's trajectory
  optimization or guarantee a new route around a live obstacle. The static
  global map does not gain live obstacles from the local layer.

Do not assume this costmap or a software watchdog is safety-rated collision
avoidance. Unobserved objects, missing scans, localization drift and stale TF can
make the displayed scene wrong.
