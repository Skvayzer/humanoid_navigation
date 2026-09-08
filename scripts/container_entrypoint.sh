#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/foxy/setup.bash
source /opt/g1_slam_ws/install/setup.bash
set -u
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-/opt/g1_slam/config/cyclonedds.xml}"
mkdir -p "$HOME" "$ROS_LOG_DIR"

mode="${1:-validate}"
shift || true

case "$mode" in
  validate)
    python3 - <<'PY'
import numpy
import scipy
import open3d
import rclpy
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"open3d={open3d.__version__}")
print("rclpy=ok")
PY
    ros2 pkg executables fast_lio
    ros2 pkg executables lidar_localization_ros2
    ros2 pkg prefix ndt_omp_ros2
    ros2 pkg prefix nav2_navfn_planner
    ros2 pkg prefix nav2_regulated_pure_pursuit_controller
    ros2 interface show livox_ros_driver2/msg/CustomMsg
    ;;
  odometry|mapping|localization)
    exec /opt/g1_slam/run_stack.sh "$mode" "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    echo "usage: $0 {validate|odometry|mapping|localization|shell}" >&2
    exit 2
    ;;
esac
