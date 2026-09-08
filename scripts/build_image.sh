#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
image="${G1_SLAM_IMAGE:-g1-slam-foxy:backup-20260713}"

if [[ ! -d "$project_dir/vendor/open3d/cpu" ]]; then
  echo "missing vendor/open3d; stage the robot's tested ARM64 Open3D package first" >&2
  exit 2
fi
for middleware_lib in \
  "$project_dir/vendor/unitree_cyclone/libddsc.so" \
  "$project_dir/vendor/unitree_cyclone/librmw_cyclonedds_cpp.so"; do
  if [[ ! -r "$middleware_lib" ]]; then
    echo "missing Unitree Cyclone runtime: $middleware_lib" >&2
    exit 2
  fi
done

exec sudo docker build \
  --label com.g1-slam.source=robotics_slam_nav_backup_20260713_111143 \
  --label com.g1-slam.role=slam-localization-navigation \
  -t "$image" -f "$project_dir/docker/Dockerfile" "$project_dir"
