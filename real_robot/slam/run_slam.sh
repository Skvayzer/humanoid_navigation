#!/usr/bin/env bash
# SLAM for the whole-body pipeline, inside the SLAM image (real_robot/docker/slam.Dockerfile).
#   run_slam.sh odom                 restamp + FAST_LIO_G1 + floor anchor (map = floor frame at the start mark)
#   run_slam.sh mapping <name>       odom + map_builder; stop (SIGINT/SIGTERM) to save /data/maps/<name>.*
#   run_slam.sh localize <name>      restamp + FAST_LIO_G1 + NDT in /data/maps/<name>.pcd from its start pose
# The robot must stand still on the start mark for the first ~5 s (IMU init, floor anchor).
set -eo pipefail
source /opt/ros/foxy/setup.bash; source /opt/g1_slam_ws/install/setup.bash
set -u
mode="${1:?odom|mapping <name>|localize <name>}"; name="${2:-}"
cfg=/opt/real_robot/config; tools=/opt/real_robot/slam
mkdir -p /data/logs /data/maps
pids=(); relay=
cleanup() {
  trap - INT TERM EXIT
  [[ -z "$relay" ]] || kill -INT "$relay" 2>/dev/null || true
  for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup INT TERM EXIT

if ros2 node list --no-daemon 2>/dev/null | grep -Eq '^/g1_slam/|/g1_sensor_restamp$'; then
  echo "refusing to start: another g1_slam / restamp node is running" >&2; exit 6
fi
for _ in $(seq 1 20); do
  [[ "$(ros2 topic type /utlidar/cloud_livox_mid360 2>/dev/null)" == sensor_msgs/msg/PointCloud2 ]] && break; sleep 1
done
[[ "$(ros2 topic type /utlidar/imu_livox_mid360 2>/dev/null)" == sensor_msgs/msg/Imu ]] || { echo "vendor LiDAR/IMU topics not visible" >&2; exit 7; }

if [[ "${G1_SKIP_RESTAMP:-0}" != 1 ]]; then      # laptop: 1 = replay the recorded /g1/imu instead
  ros2 run g1_sensor_restamp restamp_node --ros-args -r __node:=g1_sensor_restamp \
    --params-file "$(ros2 pkg prefix g1_sensor_restamp)/share/g1_sensor_restamp/config/restamp.yaml" > /data/logs/restamp.log 2>&1 &
  pids+=($!)
fi
ros2 run fast_lio fastlio_mapping --ros-args -r __node:=fast_lio -r __ns:=/g1_slam --params-file "$cfg/fastlio_g1.yaml" \
  -r /Odometry:=/g1_slam/odometry -r /path:=/g1_slam/path -r /cloud_registered:=/g1_slam/cloud_registered \
  -r /cloud_registered_body:=/g1_slam/cloud_registered_body -r /map_save:=/g1_slam/map_save > /data/logs/fast_lio.log 2>&1 &
pids+=($!)

case "$mode" in
  odom|mapping)
    : > /data/logs/floor_anchor.log      # truncate here: the background redirect below may run after the first grep
    python3 "$tools/floor_anchor.py" "/data/logs/floor_anchor_$(date -u +%Y%m%dT%H%M%SZ).json" > /data/logs/floor_anchor.log 2>&1 &
    pids+=($!)
    for _ in $(seq 1 60); do grep -q 'map -> camera_init fixed' /data/logs/floor_anchor.log 2>/dev/null && break; sleep 1; done
    grep 'map -> camera_init fixed' /data/logs/floor_anchor.log || { echo "floor anchor failed; see /data/logs/floor_anchor.log" >&2; exit 8; }
    if [[ "$mode" == mapping ]]; then
      [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "mapping needs a simple map name" >&2; exit 2; }
      [[ ! -e "/data/maps/$name.pcd" ]] || { echo "refusing to overwrite /data/maps/$name.pcd" >&2; exit 3; }
      python3 "$tools/map_builder.py" /data/maps "$name" > /data/logs/map_builder.log 2>&1 &
      mb=$!; pids+=($mb)
      echo "MAPPING $name: walk the robot slowly through the room and back to the start mark; stop to save."
    else
      echo "ODOM: map = floor frame at the start mark."
    fi ;;
  localize)
    [[ -r "/data/maps/$name.pcd" && -r "/data/maps/$name.yaml" ]] || { echo "map /data/maps/$name.{pcd,yaml} missing" >&2; exit 4; }
    ros2 run lidar_localization_ros2 lidar_localization_node --ros-args -r __node:=ndt_localizer -r __ns:=/g1_localization \
      --params-file "$cfg/ndt_g1.yaml" --params-file "/data/maps/$name.yaml" \
      -r cloud:=/g1_slam/cloud_registered_body -r odom:=/g1_slam/odometry -r initialpose:=/initialpose > /data/logs/ndt_localizer.log 2>&1 &
    pids+=($!)
    svc=/g1_localization/ndt_localizer/change_state
    for _ in $(seq 1 40); do ros2 service list --no-daemon 2>/dev/null | grep -qx "$svc" && break; sleep 0.25; done
    ros2 service call "$svc" lifecycle_msgs/srv/ChangeState '{transition: {id: 1}}' | grep -Eq 'success[=:] *(True|true)' || { echo "NDT configure failed" >&2; exit 11; }
    ros2 service call "$svc" lifecycle_msgs/srv/ChangeState '{transition: {id: 3}}' | grep -Eq 'success[=:] *(True|true)' || { echo "NDT activate failed" >&2; exit 12; }
    echo "LOCALIZE in $name from its start pose." ;;
  *) echo "usage: $0 odom | mapping <name> | localize <name>" >&2; exit 2 ;;
esac
# viewer / executor clouds; not in pids: a relay crash must not stop SLAM (map_builder publishes the map when mapping)
python3 "$tools/cloud_relay.py" $([[ "$mode" == mapping ]] && echo --no-map) > /data/logs/cloud_relay.log 2>&1 &
relay=$!
wait -n "${pids[@]}"; echo "a SLAM process exited; stopping the others (logs in /data/logs)" >&2
