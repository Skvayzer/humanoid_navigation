#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_dir="${G1_CAT_RUNTIME_DIR:-/home/unitree/g1_cat_runtime}"
nav_runtime="${G1_NAV_RUNTIME_DIR:-/home/unitree/g1_slam_runtime/nav}"
name=g1-cat-perception-preview
image=g1-cat-perception:preview
action="${1:-status}"

check_owner() {
  local role
  role="$(sudo docker inspect --format '{{index .Config.Labels "com.g1-cat.role"}}' "$name")"
  [[ "$role" == perception-only ]] || { echo "Refusing to touch non-CAT container $name" >&2; exit 3; }
}

case "$action" in
  build) exec bash "$project_dir/cat/scripts/build.sh" ;;
  validate)
    exec sudo docker run --rm --network none --read-only --cap-drop ALL \
      --security-opt no-new-privileges:true --tmpfs /tmp:rw,nosuid,nodev,size=128m \
      "$image" validate ;;
  start)
    [[ -d "$nav_runtime" ]] || { echo "Cannot verify disarmed state: missing $nav_runtime" >&2; exit 3; }
    [[ ! -e "$nav_runtime/g1_motion_armed" ]] || { echo 'Refusing: motion gateway arm token exists.' >&2; exit 3; }
    if sudo docker inspect "$name" >/dev/null 2>&1; then
      check_owner
      echo "CAT container exists. Use this script's status/logs or stop before recreating." >&2
      exit 3
    fi
    mkdir -p "$runtime_dir"
    echo 'Starting PERCEPTION ONLY. No policy, robot commands, or Nav2 changes.'
    exec sudo docker run --detach --name "$name" --network host --restart no \
      --user "$(id -u):$(id -g)" --read-only --cap-drop ALL \
      --security-opt no-new-privileges:true --pids-limit 128 --cpus 2 --memory 1536m \
      --log-opt max-size=10m --log-opt max-file=2 \
      --tmpfs /tmp:rw,nosuid,nodev,size=128m \
      --mount "type=bind,src=$runtime_dir,dst=/data" \
      --mount "type=bind,src=$nav_runtime,dst=/run/g1_nav,readonly=true" \
      --mount "type=bind,src=$project_dir/cat/config,dst=/opt/cat/config,readonly=true" \
      --mount "type=bind,src=$project_dir/config/cyclonedds.xml,dst=/opt/cat/cyclonedds.xml,readonly=true" \
      -e CYCLONEDDS_URI=/opt/cat/cyclonedds.xml \
      -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" "$image" preview ;;
  stop)
    check_owner
    mkdir -p "$runtime_dir"
    sudo docker stop --time 1 "$name"
    sudo docker logs "$name" > "$runtime_dir/last_container.log" 2>&1
    # Only this label-checked preview container is removed; snapshots are retained.
    sudo docker rm "$name"
    echo "CAT preview removed. Snapshot/log retained in $runtime_dir. SLAM/Nav2 untouched." ;;
  status)
    check_owner
    sudo docker inspect --format 'status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$name" ;;
  logs) check_owner; exec sudo docker logs --tail 30 "$name" ;;
  *) echo "Usage: $0 {build|validate|start|status|logs|stop}" >&2; exit 2 ;;
esac
