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
#   g1.sh up --geofence XMIN XMAX YMIN YMAX [executor args]   whole stack once, then drive it from the remote
#   g1.sh down                                stop all of it
#   g1.sh viz [start|stop]                    Foxglove bridge whose buttons may write /g1_wb/{operator,scene,obstacle_mode,goal_pose}
#   g1.sh record-motors [start|stop]          full-rate /lowstate + /lowcmd log (read-only)
#   g1.sh status | logs [slam|wb]
# executor args: --scene NAME (--anchor X Y YAW | --anchor-at-robot) [--geofence XMIN XMAX YMIN YMAX]
#                [--policy u596|robust_delay_v1_u150] [--mount unitree_urdf|robot_xml]
# Everything on disk stays in $G1_WORKSPACE (default: the directory that contains this checkout).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace="${G1_WORKSPACE:-$(dirname "$repo")}"
runtime="$workspace/runtime"
slam_image=g1-rr-slam:latest; wb_image=g1-rr-wholebody:latest
slam_name=g1-rr-slam; wb_name=g1-rr-wholebody
dds="$repo/config/cyclonedds.xml"
mkdir -p "$runtime/slam/maps" "$runtime/slam/logs" "$runtime/wholebody/logs" "$runtime/wholebody/scenes"
# every check below goes through sudo docker; a failed password prompt would read as "container not running"
sudo -n true 2>/dev/null || sudo -v || { echo "g1.sh needs sudo for docker: run it from a terminal (ssh -t)" >&2; exit 4; }

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

dockerfile() {   # BuildKit: as is; legacy builder (robot's Docker 24 without buildx): cache mounts stripped
  if sudo docker buildx version >/dev/null 2>&1; then echo "$1"; return; fi
  local legacy="$repo/real_robot/docker/.legacy"; mkdir -p "$legacy"
  sed 's#--mount=type=cache,target=/ccache ##' "$1" > "$legacy/$(basename "$1")"; echo "$legacy/$(basename "$1")"
}

case "${1:-status}" in
  build)
    what="${2:-all}"
    bash "$repo/scripts/fetch_runtime.sh"
    if [[ "$what" == slam || "$what" == all ]]; then
      sudo docker build --platform linux/arm64 -f "$(dockerfile "$repo/real_robot/docker/slam.Dockerfile")" --target runtime -t "$slam_image" "$repo"; fi
    if [[ "$what" == wholebody || "$what" == all ]]; then
      sudo docker build --platform linux/arm64 -f "$(dockerfile "$repo/real_robot/docker/wholebody.Dockerfile")" --target runtime -t "$wb_image" "$repo"; fi ;;
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
      --mount "type=bind,src=$runtime/wholebody/scenes,dst=/data/scenes,readonly=true" -e G1_SCENES_DIR=/data/scenes \
      "$wb_image" run "$@" "${extra[@]}" --log-dir /data/logs >/dev/null
    sleep 3; sudo docker logs "$wb_name" 2>&1 | tail -5 ;;
  viz)              # Foxglove bridge for this stack: the CAT viewer image, writable only for /g1_wb/{operator,scene,obstacle_mode,goal_pose}
    vz=g1-rr-rosbridge
    if [[ "${2:-start}" == stop ]]; then
      exists "$vz" && { sudo docker stop --time 3 "$vz" >/dev/null; sudo docker rm "$vz" >/dev/null; }; echo "viewer bridge stopped"; exit 0
    fi
    if exists g1-cat-rosbridge; then sudo docker stop --time 3 g1-cat-rosbridge >/dev/null; sudo docker rm g1-cat-rosbridge >/dev/null; fi   # same port
    exists "$vz" && ! running "$vz" && sudo docker rm "$vz" >/dev/null
    if ! exists "$vz"; then
      sudo docker run --detach --name "$vz" --network host --restart no --user "$(id -u):$(id -g)" --read-only --cap-drop ALL \
        --security-opt no-new-privileges:true --pids-limit 128 --cpus 1 --memory 768m --log-opt max-size=10m --log-opt max-file=2 \
        --tmpfs /tmp:rw,nosuid,nodev,size=64m --mount "type=bind,src=$dds,dst=/opt/cyclonedds.xml,readonly=true" \
        --mount "type=bind,src=$repo/real_robot/foxglove/rosbridge_protocol_wb.py,dst=/opt/rosbridge_overlay/python/rosbridge_library/rosbridge_protocol.py,readonly=true" \
        -e CYCLONEDDS_URI=/opt/cyclonedds.xml -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" -e ROSBRIDGE_ADDRESS=127.0.0.1 g1-cat-rosbridge:latest >/dev/null
    fi
    for _ in $(seq 1 20); do ss -lnt | grep -q "127.0.0.1:9090 " && { echo "viewer bridge on 127.0.0.1:9090 (writable: /g1_wb/operator, scene, obstacle_mode, goal_pose)"; exit 0; }; sleep 1; done
    echo "viewer bridge did not start" >&2; sudo docker logs --tail 20 "$vz" >&2; exit 5 ;;
  up)               # the whole stack once (SLAM odom, executor active in IDLE, Foxglove bridge, motor log); then the remote drives it
    shift
    printf '%s\n' "$@" | grep -q -- '--geofence' || { echo "up needs --geofence XMIN XMAX YMIN YMAX (real free floor round the start mark)" >&2; exit 2; }
    printf '%s\n' "$@" | grep -q -- '--scene' || set -- --scene flat_walk_4m --anchor-at-robot "$@"
    running "$slam_name" || "$0" slam odom
    "$0" wb active "$@"
    "$0" viz
    "$0" record-motors start || true
    cat <<'EOF'
Remote (hold 0.5 s; L2+B acts at once):
  L2+B  DAMP, from any state        L1+Y  reset: DAMP -> IDLE
  L1+A  stand: IDLE -> STAND (3 s)  L1+X  balance: STAND -> policy, goal = here (keep the rope until it stands still)
  R1+A  go: walk the scene          R1+B  stop: goal = here
Finish: g1.sh down
EOF
    ;;
  down)
    "$0" wb stop; "$0" record-motors stop; "$0" slam stop; "$0" viz stop ;;
  record-motors)    # full-rate /lowstate + /lowcmd log, read-only (never publishes) -> runtime/wholebody/motor_logs
    mr_name=g1-rr-motorlog; mr_dir="$runtime/wholebody/motor_logs"
    if [[ "${2:-start}" == stop ]]; then
      exists "$mr_name" || { echo "motor log not running"; exit 0; }
      sudo docker stop "$mr_name" >/dev/null; sudo docker logs "$mr_name" 2>&1 | tail -2; sudo docker rm "$mr_name" >/dev/null
      echo "motor log stopped: $mr_dir"; exit 0
    fi
    clear_stopped "$mr_name" wholebody wholebody
    exists "$mr_name" && { echo "motor log already running" >&2; exit 3; }
    mkdir -p "$mr_dir"
    sudo docker run --detach --name "$mr_name" "${common[@]}" --cpus 1 --memory 1g \
      --mount "type=bind,src=$repo/real_robot/wholebody/tools,dst=/opt/tools,readonly=true" \
      --mount "type=bind,src=$mr_dir,dst=/data/motor_logs" --entrypoint bash "$wb_image" \
      -c "source /opt/ros/foxy/setup.bash && source /opt/wb/messages/setup.bash && exec python3 /opt/tools/motor_recorder.py /data/motor_logs" >/dev/null
    sleep 3; sudo docker logs "$mr_name" 2>&1 | tail -2 ;;
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
      sudo docker exec "$wb_name" bash -c "source /opt/ros/foxy/setup.bash && source /opt/wb/messages/setup.bash && timeout 3 ros2 topic echo --full-length /g1_wb/status" 2>/dev/null | sed -n 's/^data: //p' | head -1; fi ;;   # Foxy: no --once
  logs)
    which="${2:-wb}"; [[ "$which" == slam ]] && { tail -n 30 "$runtime/slam/logs/"*.log; exit 0; }
    sudo docker logs --tail 40 "$wb_name" ;;
  *) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
