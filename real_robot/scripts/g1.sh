#!/usr/bin/env bash
# One entry point for the whole-body pipeline on the G1 (run on the Jetson, from the repo checkout).
#
#   g1.sh build [slam|wholebody|all]          images (robot needs internet: README "Rebuilding")
#   g1.sh load <archive.tar.gz>               images built on the laptop (laptop.sh build-robot) instead
#   g1.sh validate                            offline tests inside the executor image (no network)
#   g1.sh slam odom | mapping <name> | localize <name>      SLAM container g1-rr-slam (detached)
#   g1.sh slam stop                           stop SLAM (mapping: waits for the map to be written)
#   g1.sh wb shadow <executor args>           policy on live data, motors untouched
#   g1.sh wb active <executor args>           policy may drive the motors (gantry checklist!)
#   g1.sh wb stop                             stop the executor (it damps for 0.5 s first if active)
#   g1.sh op <stand|balance|go|stop|damp|reset|shadow|idle>   operator command to the executor
#   g1.sh status | logs [slam|wb]
# executor args: --scene NAME (--anchor X Y YAW | --anchor-at-robot) [--geofence XMIN XMAX YMIN YMAX]
#                [--policy u596|robust_delay_v1_u150] [--mount robot_xml|unitree_urdf]
# Everything on disk stays in $G1_WORKSPACE (default: the directory that contains this checkout).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace="${G1_WORKSPACE:-$(dirname "$repo")}"
runtime="$workspace/runtime"
slam_image=g1-rr-slam:latest; wb_image=g1-rr-wholebody:latest
slam_name=g1-rr-slam; wb_name=g1-rr-wholebody
dds="$repo/config/cyclonedds.xml"
mkdir -p "$runtime/slam/maps" "$runtime/slam/logs" "$runtime/wholebody/logs"

exists() { sudo docker container inspect "$1" >/dev/null 2>&1; }
running() { [[ "$(sudo docker container inspect --format '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
owned() {
  [[ "$(sudo docker container inspect --format '{{index .Config.Labels "com.g1-real-robot.role"}}' "$1")" == "$2" ]] \
    || { echo "refusing to touch $1: not a real_robot $2 container" >&2; exit 3; }
}
clear_stopped() { if exists "$1" && ! running "$1"; then owned "$1" "$2"; sudo docker logs "$1" > "$runtime/$3/logs/$1_previous.log" 2>&1 || true; sudo docker rm "$1" >/dev/null; fi; }
common=(--network host --restart no --user "$(id -u):$(id -g)" --read-only --cap-drop ALL
        --security-opt no-new-privileges:true --tmpfs /tmp:rw,nosuid,nodev,size=256m
        --mount "type=bind,src=$dds,dst=/opt/cyclonedds.xml,readonly=true"
        -e CYCLONEDDS_URI=/opt/cyclonedds.xml -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
        --log-opt max-size=20m --log-opt max-file=3)

case "${1:-status}" in
  build)
    what="${2:-all}"
    bash "$repo/scripts/fetch_runtime.sh"
    if [[ "$what" == slam || "$what" == all ]]; then
      sudo docker build --platform linux/arm64 -f "$repo/real_robot/docker/slam.Dockerfile" -t "$slam_image" "$repo"; fi
    if [[ "$what" == wholebody || "$what" == all ]]; then
      sudo docker build --platform linux/arm64 -f "$repo/real_robot/docker/wholebody.Dockerfile" --target runtime -t "$wb_image" "$repo"; fi ;;
  load)             # images built on the laptop (laptop.sh build-robot): no robot internet, no compile here
    archive="${2:?archive from laptop.sh build-robot}"
    [[ "$(uname -m)" == aarch64 ]] || { echo "load is for the robot (arm64)" >&2; exit 2; }
    sudo docker load -i "$archive"
    for i in "$slam_image" "$wb_image"; do sudo docker image inspect "$i" --format "$i: {{.Architecture}} {{.Created}}"; done ;;
  validate)
    sudo docker run --rm --network none --read-only --cap-drop ALL --tmpfs /tmp:rw,size=256m "$wb_image" validate ;;
  slam)
    sub="${2:-}"
    if [[ "$sub" == stop ]]; then
      exists "$slam_name" || { echo "SLAM not running"; exit 0; }
      owned "$slam_name" slam
      sudo docker stop --time 90 "$slam_name" >/dev/null       # SIGTERM -> map_builder saves (can take a while)
      sudo docker logs "$slam_name" > "$runtime/slam/logs/last_container.log" 2>&1 || true
      sudo docker rm "$slam_name" >/dev/null
      echo "SLAM stopped. Maps: $runtime/slam/maps  Logs: $runtime/slam/logs"; exit 0
    fi
    [[ "$sub" =~ ^(odom|mapping|localize)$ ]] || { echo "usage: g1.sh slam odom | mapping <name> | localize <name> | stop" >&2; exit 2; }
    if exists g1-cat-bridge && [[ "$(sudo docker inspect --format '{{index .Config.Cmd 0}}' g1-cat-bridge 2>/dev/null)" == run ]]; then
      echo "the CAT preview's stationary pose (g1-cat-bridge run) is publishing map->body: stop it first" >&2; exit 3; fi
    clear_stopped "$slam_name" slam slam
    exists "$slam_name" && { echo "SLAM already running ($(sudo docker inspect --format '{{.Args}}' $slam_name))" >&2; exit 3; }
    echo "Keep the robot STILL on the start mark until the floor anchor / NDT is ready."
    sudo docker run --detach --name "$slam_name" "${common[@]}" --cpus "${G1_SLAM_CPUS:-3}" --memory 4g \
      --mount "type=bind,src=$runtime/slam/maps,dst=/data/maps" --mount "type=bind,src=$runtime/slam/logs,dst=/data/logs" \
      --mount "type=bind,src=$repo/real_robot/config,dst=/opt/real_robot/config,readonly=true" \
      --mount "type=bind,src=$repo/real_robot/slam,dst=/opt/real_robot/slam,readonly=true" \
      --stop-signal SIGINT "$slam_image" "$sub" ${3:+"$3"} >/dev/null
    for _ in $(seq 1 90); do
      if sudo docker logs "$slam_name" 2>&1 | grep -qE '^(ODOM|MAPPING|LOCALIZE)'; then sudo docker logs "$slam_name" 2>&1 | tail -3; exit 0; fi
      running "$slam_name" || { sudo docker logs "$slam_name" 2>&1 | tail -20; echo "SLAM failed; logs: $runtime/slam/logs" >&2; exit 4; }
      sleep 1
    done
    echo "SLAM not ready after 90 s; see $runtime/slam/logs" >&2; exit 4 ;;
  wb)
    sub="${2:-}"; shift 2 || true
    if [[ "$sub" == stop ]]; then
      exists "$wb_name" || { echo "executor not running"; exit 0; }
      owned "$wb_name" wholebody
      sudo docker stop --time 10 "$wb_name" >/dev/null; sudo docker logs "$wb_name" > "$runtime/wholebody/logs/last_container.log" 2>&1 || true
      sudo docker rm "$wb_name" >/dev/null; echo "executor stopped; logs in $runtime/wholebody/logs"; exit 0
    fi
    [[ "$sub" == shadow || "$sub" == active ]] || { echo "usage: g1.sh wb shadow|active <executor args> | stop" >&2; exit 2; }
    running "$slam_name" || { echo "start SLAM first (g1.sh slam odom | localize <map>)" >&2; exit 3; }
    clear_stopped "$wb_name" wholebody wholebody
    exists "$wb_name" && { echo "executor already running" >&2; exit 3; }
    extra=(--start shadow)
    if [[ "$sub" == active ]]; then
      printf '%s\n' "$@" | grep -q -- '--geofence' || { echo "active mode needs --geofence (real free floor in the map)" >&2; exit 2; }
      echo "ACTIVE: the policy may drive all 29 joints. Robot on the gantry, E-stop in hand, area clear."
      read -r -p "Type ACTIVE to continue: " answer; [[ "$answer" == ACTIVE ]] || { echo aborted; exit 1; }
      extra=(--active --start idle)
    fi
    sudo docker run --detach --name "$wb_name" "${common[@]}" --cpus "${G1_WB_CPUS:-3}" ${G1_WB_CPUSET:+--cpuset-cpus "$G1_WB_CPUSET"} \
      --memory 3g --shm-size 256m --mount "type=bind,src=$runtime/wholebody/logs,dst=/data/logs" \
      "$wb_image" run "$@" "${extra[@]}" --log-dir /data/logs >/dev/null
    sleep 3; sudo docker logs "$wb_name" 2>&1 | tail -5 ;;
  op)
    cmd="${2:?operator command}"
    [[ "$cmd" =~ ^(stand|balance|go|stop|damp|reset|shadow|idle)$ ]] || { echo "unknown command $cmd" >&2; exit 2; }
    running "$wb_name" || { echo "executor not running" >&2; exit 3; }
    sudo docker exec "$wb_name" bash -c "source /opt/ros/foxy/setup.bash && ros2 topic pub --once /g1_wb/operator std_msgs/msg/String '{data: $cmd}'" >/dev/null
    sleep 0.5; sudo docker logs --tail 3 "$wb_name" 2>&1 ;;
  status)
    for c in "$slam_name" "$wb_name" g1-cat-bridge g1-cat-perception-preview g1-cat-rosbridge; do
      exists "$c" && echo "$c: $(sudo docker inspect --format '{{.State.Status}} {{.Args}}' "$c")" || echo "$c: -"; done
    if running "$wb_name"; then
      sudo docker exec "$wb_name" bash -c "source /opt/ros/foxy/setup.bash && source /opt/wb/messages/setup.bash && timeout 3 ros2 topic echo --once /g1_wb/status" 2>/dev/null | sed -n 's/^data: //p'; fi ;;
  logs)
    which="${2:-wb}"; [[ "$which" == slam ]] && { tail -n 30 "$runtime/slam/logs/"*.log; exit 0; }
    sudo docker logs --tail 40 "$wb_name" ;;
  *) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
