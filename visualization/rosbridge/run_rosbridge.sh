#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"

# Use the exact DDS runtime used by the G1 Livox driver and SLAM container.
set +u
source /opt/ros/foxy/setup.bash
source "$G1_CYCLONE_WS/install/setup.bash"
source "$G1_LIVOX_WS/install/setup.bash"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="$G1_HOST_CYCLONEDDS_URI"
export ROS_LOG_DIR="$G1_SLAM_RUNTIME_DIR/logs/rosbridge/ros"
export PYTHONPATH="$G1_SLAM_PROJECT_DIR/visualization/rosbridge/python:${PYTHONPATH:-}"
mkdir -p "$ROS_LOG_DIR"

exec ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
  port:=9090 \
  address:=0.0.0.0 \
  'topics_glob:="[*]"' \
  'services_glob:="[/rosapi/*]"' \
  'params_glob:="[/__disabled__]"'
