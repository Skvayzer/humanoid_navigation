#!/usr/bin/env bash
# Sourced by this project's launchers only; never changes global ROS setup.
export G1_SLAM_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$G1_SLAM_PROJECT_DIR/config/site.env" ]]; then
  source "$G1_SLAM_PROJECT_DIR/config/site.env"
fi
export G1_SLAM_RUNTIME_DIR="${G1_SLAM_RUNTIME_DIR:-$HOME/g1_slam_runtime}"
export G1_SLAM_CONTAINER="${G1_SLAM_CONTAINER:-g1-slam-backup-20260713}"
export G1_SLAM_IMAGE="${G1_SLAM_IMAGE:-g1-slam-foxy:backup-20260713}"
export G1_CYCLONE_WS="${G1_CYCLONE_WS:-$HOME/cyclonedds_ws}"
export G1_LIVOX_WS="${G1_LIVOX_WS:-$HOME/livox_ws}"
export G1_HOST_CYCLONEDDS_URI="${G1_HOST_CYCLONEDDS_URI:-$G1_CYCLONE_WS/cyclonedds.xml}"
export G1_SDK_PATH="${G1_SDK_PATH:-$G1_SLAM_PROJECT_DIR/third_party/unitree_sdk2_python}"
export G1_SDK_INTERFACE="${G1_SDK_INTERFACE:-eth0}"
export G1_MOTION_PYTHON="${G1_MOTION_PYTHON:-python3}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
