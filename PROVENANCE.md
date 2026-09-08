# Deployed snapshot provenance

## Source of truth

This repository was copied read-only from the G1 on **2026-09-08**:

- `~/g1_slam_deploy`: container, algorithms, configuration, launchers and gateway.
- `~/bin`: the operator shortcuts actually used on the robot.
- `~/rosbridge_runtime/python` and launcher: working Foxglove/rosbridge patches.
- `~/livox_viz_ws/src` and launcher: standard point-cloud/IMU visualization.
- `~/unitree_sdk2_python`: clean upstream checkout at
  `a035adeaa6f8ea171bef9a43e8477abb87a0b35e`, SDK version 1.0.1.
- `~/cyclonedds_ws/src`: accompanying middleware source in the release archive.

No robot code, package installation, container, mode or process was changed while
making this repository. Public packaging edits were made only in a fresh desktop
checkout.

The original tested X2 archive was
`robotics_slam_nav_backup_20260713_111143.zip`, SHA-256
`a6edefc5bc400066f05c2ac5a76cac227a78bd2232d5f13f6058e7aeadb286a4`.
It supplied Fast-LIO and the legacy ICP localizer/floor code. NDT-OMP and
`lidar_localization_ros2` were subsequently brought into the deployed G1 stack
from the NDT work associated with `slam.zip`. The old statement that this image
contains no source from that tree is no longer correct.

## What was preserved

`manifests/robot_snapshot.sha256` records the copied files before packaging edits.
The regression check requires all captured estimator sources, YAML parameters
and navigation runtime algorithms to remain byte-identical, except the map-path
configuration in the navigation launch file. Upstream example PCD data is excluded
from public Git. The manifest is historical, not a whole-tree verification command
for the final packaged checkout.

Working fixes retained include the Livox cloud/IMU roll correction, Foxy PCL/tf2
compatibility patches, NDT absolute odometry prediction and scan-time pose/TF
publication, map floor alignment, planar base-axis handling, path frame IDs,
heading-first RPP turning, and the rosbridge schemas/ROS 2 detection/executor fixes.
The runtime localizer is NDT; `src/localizer.py` is the retained legacy ICP option.

## Packaging-only changes

- Configurable user/workspace/runtime paths via `scripts/env.sh` and ignored
  `config/site.env`; configurable map filenames and resource ceilings.
- Motion gateway filesystem and interface settings made configurable; command
  limits, arming logic and control algorithm preserved.
- Preview shortcut explicitly disarms; generic SLAM stop delegates to the
  mode-aware save-before-removal script. These are safety-related wrapper changes.
- New idempotent shortcut installer refuses conflicts. The old destructive
  fixed-path rsync deployment script is disabled.
- Updated status text, documentation, source checks, CI and dependency fetching.
- Read/write mounts use the invoking account's UID/GID; image labels now describe
  SLAM + localization + navigation.
- No algorithm retuning or software deployment back to the existing robot.

## Runtime dependencies

The release `robot-snapshot-20260908` contains:

- `g1-foxy-arm64-runtime-20260908.tar.gz`: exact deployed Open3D 0.16.0
  CPython 3.8 ARM64 CPU extension, `libddsc.so`, and
  `librmw_cyclonedds_cpp.so`. The headless Python initializer is in Git.
- `g1-cyclonedds-source-20260908.tar.gz`: source copied from the same robot's
  CycloneDDS 0.10.2 and RMW workspace, with original licenses/notices.

SHA-256 manifests pin both archives and each runtime binary. GUI resources/driver
libraries from the installed Open3D package are not used by this stack and are not
redistributed. The native extension is unchanged.

Middleware source baseline: CycloneDDS directory `cyclonedds-0.10.2` had no Git
metadata. RMW's upstream HEAD was
`c12abc56983204f1d91f2d839d394528c7b29b42` from
https://github.com/ros2/rmw_cyclonedds, with local modifications: the **actual
worktree**, not an unmodified upstream checkout, is archived. Historical build
flags are not fully captured, so the binary hashes—not a claim of a
byte-identical source rebuild—identify the tested runtime.

Open3D upstream: https://github.com/isl-org/Open3D/tree/v0.16.0.
Unitree SDK upstream:
https://github.com/unitreerobotics/unitree_sdk2_python/tree/a035adeaa6f8ea171bef9a43e8477abb87a0b35e.
rosbridge baseline: https://github.com/RobotWebTools/rosbridge_suite/tree/1.3.1;
robot Debian package versions were rosapi 1.3.1-1focal.20230527.072758,
rosbridge_library 1.3.1-1focal.20230527.071226, and
rosbridge_server 1.3.1-1focal.20230527.072934.

This source snapshot is not an export of the old Docker image. Docker's base tag
and apt repositories are external rebuild dependencies. Preserve a validated
image digest/package inventory if a byte-identical long-term deployment is needed.
