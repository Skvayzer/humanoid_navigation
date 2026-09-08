#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"

set +u
source /opt/ros/foxy/setup.bash
source "$G1_CYCLONE_WS/install/setup.bash"
source "$G1_LIVOX_WS/install/setup.bash"
source "$G1_SLAM_PROJECT_DIR/visualization/livox/install/setup.bash"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="$G1_HOST_CYCLONEDDS_URI"
export ROS_LOG_DIR="$G1_SLAM_RUNTIME_DIR/logs/livox_viz/ros"

mkdir -p "$ROS_LOG_DIR"
exec ros2 run livox_viz_bridge livox_viz_node --ros-args \
  -r __node:=livox_viz_bridge \
  -p lidar_input:=/livox/lidar \
  -p imu_input:=/livox/imu \
  -p cloud_output:=/g1_viz/livox_points \
  -p marker_output:=/g1_viz/livox_imu_markers \
  -p scan_period_ms:=100 \
  -p max_points_per_scan:=200000
