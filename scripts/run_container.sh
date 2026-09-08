#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

mode="${1:-validate}"
shift || true
image="${G1_SLAM_IMAGE:-g1-slam-foxy:backup-20260713}"
name="$G1_SLAM_CONTAINER"
runtime_dir="$G1_SLAM_RUNTIME_DIR"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remove_args=()
if [[ "$mode" == validate ]]; then
  remove_args=(--rm)
fi
detach_args=()
if [[ "${G1_SLAM_DETACH:-0}" == 1 && "$mode" != validate ]]; then
  detach_args=(--detach)
fi

if sudo docker container inspect "$name" >/dev/null 2>&1; then
  echo "container already exists: $name" >&2
  echo "use scripts/stop_container.sh before creating a replacement" >&2
  exit 3
fi

exec sudo docker run --name "$name" \
  "${remove_args[@]}" \
  "${detach_args[@]}" \
  --network host \
  --user "$(id -u):$(id -g)" \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 512 \
  --memory "${G1_SLAM_MEMORY_LIMIT:-10g}" \
  --cpus "${G1_SLAM_CPU_LIMIT:-6}" \
  --shm-size 1g \
  --restart no \
  --tmpfs /tmp:rw,nosuid,nodev,size=512m \
  --mount "type=bind,src=$project_dir/config,dst=/opt/g1_slam/config,readonly=true" \
  --mount "type=bind,src=$project_dir/nav,dst=/opt/g1_slam/nav,readonly=true" \
  --mount "type=bind,src=$runtime_dir/maps,dst=/data/maps" \
  --mount "type=bind,src=$runtime_dir/logs,dst=/data/logs" \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  -e G1_NAV_MAP_YAML="${G1_NAV_MAP_YAML:-/data/maps/teaching_lab_nav/teaching_lab_nav.yaml}" \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI=/opt/g1_slam/config/cyclonedds.xml \
  -e RCUTILS_LOGGING_BUFFERED_STREAM=1 \
  "$image" "$mode" "$@"
