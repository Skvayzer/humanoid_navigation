#!/usr/bin/env bash
# Laptop (amd64) development: the robot's images, a recorded bag instead of the robot, MuJoCo instead of the G1.
# DDS stays on loopback (config/cyclonedds_laptop.xml) in domain $LAPTOP_DOMAIN (default 77).
#
#   laptop.sh build                          SLAM + executor(dev, with MuJoCo) images for amd64
#   laptop.sh build-robot                    arm64 images for the Jetson (QEMU), validated, saved for g1.sh load
#   laptop.sh validate                       executor image tests
#   laptop.sh map <bag_dir> <name>           FAST_LIO_G1 mapping on a bag -> $OUT/maps/<name>.*
#   laptop.sh localize <bag_dir> <name>      NDT localisation in <name> on a bag -> trajectory CSVs
#   laptop.sh lio <bag_dir> <tag> [g1imu]    odometry only; with g1imu: use the bag's recorded /g1/imu instead of our restamper
#   laptop.sh sim [executor args]            simulated G1 (gantry) + executor, both over ROS; operator via 'laptop.sh op'
#   laptop.sh op <command>                   operator command to the simulated executor
#   laptop.sh down                           stop everything started here
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${OUT:-$(cd "$repo/../.." && pwd)/results/laptop}"
domain="${LAPTOP_DOMAIN:-77}"
slam_image=g1-rr-slam:amd64; wb_image=g1-rr-wholebody:amd64-dev
mkdir -p "$out/maps" "$out/logs" "$out/traj"
dds=(--mount "type=bind,src=$repo/real_robot/config/cyclonedds_laptop.xml,dst=/opt/dds.xml,readonly=true"
     -e CYCLONEDDS_URI=/opt/dds.xml -e ROS_DOMAIN_ID="$domain")
keep=(--network host --user "$(id -u):$(id -g)" "${dds[@]}")      # long-lived containers (stopped explicitly)
base=(--rm "${keep[@]}")
slam_mounts=(--mount "type=bind,src=$out/maps,dst=/data/maps" --mount "type=bind,src=$out/logs,dst=/data/logs"
             --mount "type=bind,src=$repo/real_robot/config,dst=/opt/real_robot/config,readonly=true"
             --mount "type=bind,src=$repo/real_robot/slam,dst=/opt/real_robot/slam,readonly=true")

play() {   # play <bag_dir> <name> <topics...>: bag player container (detached), on a working copy of the bag
  local bag="$1" name="$2"; shift 2
  local cache="$out/bag_cache/$(basename "$(cd "$bag" && pwd)")"   # rosbag2 (Foxy) cannot open sqlite read-only
  [[ -f "$cache/bag/metadata.yaml" ]] || { mkdir -p "$cache"; cp -r "$bag/bag" "$cache/"; }
  docker run -d --name "$name" "${base[@]}" --mount "type=bind,src=$cache,dst=/bag" \
    --entrypoint bash "$slam_image" -c "source /opt/ros/foxy/setup.bash && sleep 4 && ros2 bag play /bag/bag --topics $*" >/dev/null
}
record() { # record <name> <csv> : odometry + map->camera_init TF to CSV (python inside the SLAM image)
  docker run -d --name "$1" "${base[@]}" --mount "type=bind,src=$out/traj,dst=/traj" --entrypoint bash "$slam_image" -c \
    "source /opt/ros/foxy/setup.bash && source /opt/g1_slam_ws/install/setup.bash && python3 /opt/real_robot/slam/record_traj.py /traj/$2" >/dev/null
}
wait_exit() { docker wait "$1" >/dev/null 2>&1 || true; }

case "${1:-}" in
  build)
    docker build --platform linux/amd64 -f "$repo/real_robot/docker/slam.Dockerfile" --target dev -t "$slam_image" "$repo"
    docker build --platform linux/amd64 -f "$repo/real_robot/docker/wholebody.Dockerfile" --target dev -t "$wb_image" "$repo" ;;
  build-robot)      # arm64 images for the Jetson under QEMU (qemu-user-static), saved for g1.sh load
    bash "$repo/scripts/fetch_runtime.sh"
    # both images at once: the SLAM build sits on one huge translation unit most of the time
    docker build --platform linux/arm64 --build-arg MAKE_JOBS="${MAKE_JOBS:-6}" --target runtime \
      -f "$repo/real_robot/docker/slam.Dockerfile" -t g1-rr-slam:latest "$repo" > "$out/logs/build_robot_slam.log" 2>&1 & slam_pid=$!
    docker build --platform linux/arm64 -f "$repo/real_robot/docker/wholebody.Dockerfile" --target runtime \
      -t g1-rr-wholebody:latest "$repo" > "$out/logs/build_robot_wholebody.log" 2>&1 & wb_pid=$!
    wait $wb_pid || { echo "executor image failed: $out/logs/build_robot_wholebody.log" >&2; kill $slam_pid 2>/dev/null; exit 1; }
    wait $slam_pid || { echo "SLAM image failed: $out/logs/build_robot_slam.log" >&2; exit 1; }
    echo "both arm64 images built"
    docker run --rm --platform linux/arm64 --network none -e G1_TIMING_TESTS=0 g1-rr-wholebody:latest validate   # timing checks run on the robot
    mkdir -p "$out/images"
    docker save g1-rr-slam:latest g1-rr-wholebody:latest | gzip -1 > "$out/images/g1_rr_images_arm64.tar.gz"
    ls -la "$out/images" ;;
  validate) docker run --rm --network none "$wb_image" validate ;;
  map|localize|lio)
    bag="${2:?bag dir}"; name="${3:?name}"; topics="/utlidar/cloud_livox_mid360 /utlidar/imu_livox_mid360"
    mode="$1"; [[ "$mode" == map ]] && mode=mapping; [[ "$mode" == lio ]] && mode=odom
    slam_args=("$mode"); [[ "$mode" != odom ]] && slam_args+=("$name")
    env_extra=()
    if [[ "${4:-}" == g1imu ]]; then topics="/utlidar/cloud_livox_mid360 /utlidar/imu_livox_mid360 /g1/imu"; env_extra=(-e G1_SKIP_RESTAMP=1); fi
    docker rm -f lap-slam lap-play lap-rec >/dev/null 2>&1 || true
    docker run -d --name lap-slam "${keep[@]}" "${slam_mounts[@]}" "${env_extra[@]}" --stop-signal SIGINT "$slam_image" "${slam_args[@]}" >/dev/null
    record lap-rec "${name}_${1}${4:+_$4}.csv"
    play "$bag" lap-play $topics
    wait_exit lap-play
    sleep 3; docker stop --time 120 lap-slam >/dev/null; docker stop lap-rec >/dev/null 2>&1 || true
    docker logs lap-slam > "$out/logs/${name}_${1}_container.log" 2>&1; docker rm lap-slam >/dev/null
    echo "done: maps $out/maps, trajectories $out/traj, logs $out/logs" ;;
  sim)
    shift
    docker rm -f lap-fake lap-wb >/dev/null 2>&1 || true
    docker run -d --name lap-fake "${keep[@]}" "$wb_image" fake --gantry --scene "${SCENE:-sidegap_corridor_048}" ${FAKE_ARGS:-} >/dev/null
    sleep 3
    docker run -d --name lap-wb "${keep[@]}" --shm-size 256m --mount "type=bind,src=$out/logs,dst=/data/logs" "$wb_image" run \
      --scene "${SCENE:-sidegap_corridor_048}" --anchor-at-robot --geofence-from-scene 1.0 --active --log-dir /data/logs "$@" >/dev/null
    sleep 5; docker logs lap-wb 2>&1 | tail -5
    echo "operator: laptop.sh op stand | balance | go | damp ...   rope: laptop.sh rope 0.05 | 9 ;  buttons: laptop.sh buttons L2,B" ;;
  op) docker exec lap-wb bash -c "source /opt/ros/foxy/setup.bash && ros2 topic pub --once /g1_wb/operator std_msgs/msg/String '{data: $2}'" >/dev/null; docker logs --tail 3 lap-wb ;;
  rope) docker exec lap-fake bash -c "source /opt/ros/foxy/setup.bash && ros2 topic pub --once /g1_fake/rope std_msgs/msg/Float32 '{data: $2}'" >/dev/null ;;
  buttons) docker exec lap-fake bash -c "source /opt/ros/foxy/setup.bash && ros2 topic pub --once /g1_fake/buttons std_msgs/msg/String '{data: \"$2\"}'" >/dev/null ;;
  status) docker exec lap-wb bash -c "source /opt/ros/foxy/setup.bash && timeout 3 ros2 topic echo --once /g1_wb/status" | sed -n 's/^data: //p' ;;
  down) docker rm -f lap-slam lap-play lap-rec lap-fake lap-wb >/dev/null 2>&1 || true ;;
  *) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
