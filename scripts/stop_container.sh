#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

name="$G1_SLAM_CONTAINER"
runtime_dir="$G1_SLAM_RUNTIME_DIR"

if ! sudo docker container inspect "$name" >/dev/null 2>&1; then
  echo "SLAM container does not exist: $name"
  exit 0
fi

state="$(sudo docker inspect --format '{{.State.Status}}' "$name")"
mode="$(sudo docker inspect --format '{{index .Config.Cmd 0}}' "$name")"

if [[ "$state" == "running" && "$mode" == "mapping" ]]; then
  mkdir -p "$runtime_dir/logs"
  echo "Saving the map before immediate container removal..."
  save_output="$(sudo docker exec "$name" bash -lc '
    set +u
    source /opt/ros/foxy/setup.bash
    source /opt/g1_slam_ws/install/setup.bash
    set -u
    timeout 60 ros2 service call /g1_slam/map_save std_srvs/srv/Trigger "{}"
  ')"
  printf '%s\n' "$save_output" | tee -a "$runtime_dir/logs/map_save.log"
  if ! grep -q 'success=True' <<<"$save_output"; then
    echo "Map save did not report success; leaving the container running." >&2
    exit 5
  fi
fi

# The state worth preserving is bind-mounted under runtime_dir. Once an
# explicit mapping save has completed, no Docker grace-period wait is needed.
sudo docker rm -f "$name"
