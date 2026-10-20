#!/usr/bin/env bash
# CAT perception preview (no motion) on a G1 with vendor /utlidar LiDAR topics.
# Runs two containers: g1-cat-bridge (LiDAR adapter + measured stationary pose)
# the repo's unmodified CAT container in research mode (ESDF/SDF fields), and
# g1-cat-rosbridge for Foxglove (robot localhost:9090 by default).
# Everything on disk stays under the workspace folder that contains this module.
set -euo pipefail
module_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="${G1_CAT_REPO:-$(cd "$module_dir/../.." && pwd)}"          # humanoid_navigation checkout (feature/g1-real-robot)
workspace="${G1_WORKSPACE:-$(dirname "$repo")}"                  # ~/workspace/artemvoronov on the robot
export G1_CAT_RUNTIME_DIR="${G1_CAT_RUNTIME_DIR:-$workspace/runtime/cat}"
logs="$workspace/logs"
bridge_name=g1-cat-bridge
bridge_image=g1-cat-bridge:latest
mkdir -p "$logs" "$G1_CAT_RUNTIME_DIR"

bridge_exists() { sudo docker container inspect "$bridge_name" >/dev/null 2>&1; }
viz_name=g1-cat-rosbridge
viz_image=g1-cat-rosbridge:latest
owned() {  # owned <container> <expected role label>
  [[ "$(sudo docker container inspect --format '{{index .Config.Labels "com.g1-cat-bridge.role"}}' "$1")" == "$2" ]] \
    || { echo "Refusing to touch container $1 (not created by this module)" >&2; exit 3; }
}
bridge_owned() { owned "$bridge_name" read-only-sensor-adapter; }
running() { [[ "$(sudo docker container inspect --format '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
clear_exited() {  # clear_exited <container> <role>; removes only OUR stopped containers (e.g. after a reboot)
  if sudo docker container inspect "$1" >/dev/null 2>&1 && ! running "$1"; then
    owned "$1" "$2"; sudo docker logs "$1" > "$logs/$1_previous.log" 2>&1 || true
    sudo docker rm "$1" >/dev/null && echo "Removed stopped $1 (log: $logs/$1_previous.log)"
  fi
}
clear_exited_cat() {
  if sudo docker container inspect g1-cat-perception-preview >/dev/null 2>&1 && ! running g1-cat-perception-preview; then
    cat_cmd stop; fi
}

build_bridge() {
  local ctx="$module_dir/.build_context"
  rm -rf "$ctx"; mkdir -p "$ctx"
  cp -r "$module_dir/ros/g1_cat_bridge" "$repo/src/livox_ros_driver2" "$ctx/"
  mkdir -p "$ctx/unitree_cyclone"
  cp "$repo/vendor/unitree_cyclone/libddsc.so" "$repo/vendor/unitree_cyclone/librmw_cyclonedds_cpp.so" "$ctx/unitree_cyclone/"
  cp "$module_dir/docker/entrypoint.sh" "$ctx/"
  cp -r "$repo/visualization/rosbridge/python" "$ctx/rosbridge_overlay"
  cp "$module_dir/docker/rosbridge_entrypoint.sh" "$ctx/"
  sudo docker build -f "$module_dir/docker/Dockerfile" -t "$bridge_image" "$ctx"
  sudo docker build -f "$module_dir/docker/rosbridge.Dockerfile" -t "$viz_image" "$ctx"
  rm -rf "$ctx"
}

start_viz() {
  clear_exited "$viz_name" rosbridge-viz
  if sudo docker container inspect "$viz_name" >/dev/null 2>&1; then owned "$viz_name" rosbridge-viz; echo "$viz_name already running"; return 0; fi
  sudo docker run --detach --name "$viz_name" --network host --restart no \
    --user "$(id -u):$(id -g)" --read-only --cap-drop ALL --security-opt no-new-privileges:true \
    --pids-limit 128 --cpus 1 --memory 768m --log-opt max-size=10m --log-opt max-file=2 \
    --tmpfs /tmp:rw,nosuid,nodev,size=64m \
    --mount "type=bind,src=$repo/config/cyclonedds.xml,dst=/opt/cyclonedds.xml,readonly=true" \
    -e CYCLONEDDS_URI=/opt/cyclonedds.xml -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -e ROSBRIDGE_ADDRESS="${ROSBRIDGE_ADDRESS:-127.0.0.1}" "$viz_image" >/dev/null
  for _ in $(seq 1 20); do
    if ss -lnt | grep -q ":9090 "; then echo "rosbridge listening on ${ROSBRIDGE_ADDRESS:-127.0.0.1}:9090"; return 0; fi
    sleep 1
  done
  echo "rosbridge did not start. Logs:" >&2; sudo docker logs --tail 30 "$viz_name" >&2; exit 5
}

stop_viz() {
  sudo docker container inspect "$viz_name" >/dev/null 2>&1 || return 0
  owned "$viz_name" rosbridge-viz
  sudo docker stop --time 3 "$viz_name" >/dev/null || true
  sudo docker rm "$viz_name" >/dev/null
  echo "rosbridge stopped."
}

start_bridge() {   # start_bridge [run|converter]
  local bridge_mode="${1:-run}"
  clear_exited "$bridge_name" read-only-sensor-adapter
  if [[ "$bridge_mode" == run ]] && sudo docker container inspect g1-rr-slam >/dev/null 2>&1; then
    echo "SLAM (g1-rr-slam) is running: its TF map->camera_init->body would conflict with the stationary pose." >&2
    echo "Use 'perception.sh start-with-slam' instead." >&2; exit 3
  fi
  if bridge_exists; then bridge_owned; echo "$bridge_name already exists; use stop first" >&2; exit 3; fi
  sudo docker run --detach --name "$bridge_name" --network host --restart no \
    --user "$(id -u):$(id -g)" --read-only --cap-drop ALL --security-opt no-new-privileges:true \
    --pids-limit 64 --cpus 2 --memory 512m --log-opt max-size=10m --log-opt max-file=2 \
    --tmpfs /tmp:rw,nosuid,nodev,size=64m \
    --mount "type=bind,src=$repo/config/cyclonedds.xml,dst=/opt/cyclonedds.xml,readonly=true" \
    -e CYCLONEDDS_URI=/opt/cyclonedds.xml -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" "$bridge_image" "$bridge_mode" >/dev/null
  if [[ "$bridge_mode" == converter ]]; then echo "Bridge started (LiDAR converter only; pose from SLAM)."; return 0; fi
  echo "Bridge started. Waiting for the measured stationary pose (keep the robot still) ..."
  for _ in $(seq 1 40); do
    if sudo docker logs "$bridge_name" 2>&1 | grep -q "Pose fixed"; then
      sudo docker logs "$bridge_name" 2>&1 | grep -E "READ-ONLY|Pose fixed" | sed 's/^/  /'; return 0; fi
    [[ "$(sudo docker container inspect --format '{{.State.Running}}' "$bridge_name")" == true ]] || break
    sleep 1
  done
  echo "Bridge did not report a fixed pose. Logs:" >&2; sudo docker logs --tail 30 "$bridge_name" >&2; exit 4
}

stop_bridge() {
  bridge_exists || return 0
  bridge_owned
  sudo docker stop --time 3 "$bridge_name" >/dev/null || true
  sudo docker logs "$bridge_name" > "$logs/bridge_last.log" 2>&1 || true
  sudo docker rm "$bridge_name" >/dev/null
  echo "Bridge stopped (log saved to $logs/bridge_last.log)."
}

cat_cmd() { bash "$repo/cat/scripts/container.sh" "$@"; }

case "${1:-status}" in
  build)    echo "== [1/2] bridge + rosbridge images =="; build_bridge
            echo "== [2/2] CAT image (repo cat/scripts/build.sh) =="; bash "$repo/cat/scripts/build.sh"
            echo "== CAT preview images built (SLAM: real_robot/scripts/g1.sh build slam) ==" ;;
  validate) cat_cmd validate ;;
  start)    clear_exited_cat; start_bridge; cat_cmd start-research; start_viz ;;
  start-with-slam)                    # CAT ESDF while walking: TF from real_robot SLAM (g1.sh slam ...)
            sudo docker container inspect g1-rr-slam >/dev/null 2>&1 || { echo "start SLAM first: real_robot/scripts/g1.sh slam odom" >&2; exit 3; }
            clear_exited_cat; start_bridge converter; cat_cmd start-research; start_viz ;;
  start-preview) clear_exited_cat; start_bridge; cat_cmd start; start_viz ;;
  viz-start) start_viz ;;
  viz-stop)  stop_viz ;;
  stop)     stop_viz; if sudo docker container inspect g1-cat-perception-preview >/dev/null 2>&1; then cat_cmd stop; fi; stop_bridge ;;
  status)   bridge_exists && sudo docker container inspect --format "bridge: {{.State.Status}} since {{.State.StartedAt}}" "$bridge_name" || echo "bridge: not running"
            sudo docker container inspect g1-cat-perception-preview >/dev/null 2>&1 && echo "cat: $(cat_cmd status)" || echo "cat: not running"
            sudo docker container inspect "$viz_name" >/dev/null 2>&1 && sudo docker container inspect --format "rosbridge: {{.State.Status}}" "$viz_name" || echo "rosbridge: not running" ;;
  logs)     bridge_exists && { echo "== bridge =="; sudo docker logs --tail 15 "$bridge_name" 2>&1; }
            sudo docker container inspect g1-cat-perception-preview >/dev/null 2>&1 && { echo "== cat =="; cat_cmd logs; } ;;
  *) echo "usage: $0 {build|validate|start|start-with-slam|start-preview|viz-start|viz-stop|stop|status|logs}" >&2; exit 2 ;;
esac
