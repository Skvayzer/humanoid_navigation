#!/usr/bin/env bash
# SLAM for the whole-body pipeline, inside the SLAM image (real_robot/docker/slam.Dockerfile): FAST_LIO_G1 (IMU restamp,
# odometry, localization on a saved .vmap) + the floor anchor (map -> slam_map) + the clouds for the viewer/executor.
#   run_slam.sh odom                 SLAM; map = floor frame at the start mark (measured)
#   run_slam.sh mapping <name>       odom + map_builder; stop (SIGINT/SIGTERM) to save /data/maps/<name>.vmap (FAST-LIO)
#                                    and <name>{_raw.pcd,.pcd,_path.csv,_info.json} (map_builder); then g1.sh static-esdf <name>
#   run_slam.sh localize <name>      relocalization on /data/maps/<name>.vmap anywhere on the map (no start mark);
#                                    map = that map's floor frame (<name>_floor.json from g1.sh static-esdf)
# The robot must stand still for the first ~5 s (IMU init 3 s, floor anchor).
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

lio=(--ros-args -r __node:=fast_lio -r __ns:=/g1_slam --params-file "$cfg/fastlio_g1.yaml" -r tf:=/tf -r tf_static:=/tf_static
     -r /trajectory:=/g1_slam/path)
case "$mode" in
  mapping)
    [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "mapping needs a simple map name" >&2; exit 2; }
    [[ ! -e "/data/maps/$name.vmap" && ! -e "/data/maps/$name.pcd" ]] || { echo "refusing to overwrite /data/maps/$name.*" >&2; exit 3; }
    lio+=(-p map_save:=true -p "map_file_path:=/data/maps/$name") ;;
  localize)
    for f in "$name.vmap" "${name}_floor.json"; do
      [[ -r "/data/maps/$f" ]] || { echo "/data/maps/$f missing (localize needs a FAST_LIO_G1 map + g1.sh static-esdf $name)" >&2; exit 4; }
    done
    lio+=(-p "localization.map_file:=/data/maps/$name.vmap") ;;
  odom) ;;
  *) echo "usage: $0 odom | mapping <name> | localize <name>" >&2; exit 2 ;;
esac
# the binary itself, not `ros2 run`: Foxy's ros2 run does not pass SIGINT on, and FAST-LIO writes the map (mapping)
# only when it gets one; through the wrapper it was killed at the docker stop timeout with nothing written
: > /data/logs/fast_lio.log         # truncate here: the background redirect may run after the first grep below
"$(ros2 pkg prefix fast_lio)/lib/fast_lio/fastlio_mapping" "${lio[@]}" >> /data/logs/fast_lio.log 2>&1 &
pids+=($!)

: > /data/logs/floor_anchor.log      # truncate here: the background redirect below may run after the first grep
if [[ "$mode" == localize ]]; then
  python3 "$tools/floor_anchor.py" --from-json "/data/maps/${name}_floor.json" > /data/logs/floor_anchor.log 2>&1 &
  pids+=($!)
  for _ in $(seq 1 60); do grep -q 'Localization: localized' /data/logs/fast_lio.log 2>/dev/null && break; sleep 1; done
  grep 'Localization: ' /data/logs/fast_lio.log | tail -1
  grep -q 'Localization: localized' /data/logs/fast_lio.log || { echo "not localized after 60 s: set the pose in Foxglove (3D panel, pose estimate, display frame map) or move the robot; see /data/logs/fast_lio.log" >&2; }
  echo "LOCALIZE in $name."
else
  python3 "$tools/floor_anchor.py" "/data/logs/floor_anchor_$(date -u +%Y%m%dT%H%M%SZ).json" > /data/logs/floor_anchor.log 2>&1 &
  pids+=($!)
  for _ in $(seq 1 60); do grep -q 'map -> slam_map fixed' /data/logs/floor_anchor.log 2>/dev/null && break; sleep 1; done
  grep 'map -> slam_map fixed' /data/logs/floor_anchor.log || { echo "floor anchor failed; see /data/logs/floor_anchor.log" >&2; exit 8; }
  if [[ "$mode" == mapping ]]; then
    python3 "$tools/map_builder.py" /data/maps "$name" > /data/logs/map_builder.log 2>&1 &
    pids+=($!)
    echo "MAPPING $name: walk the robot slowly through the room and back to the start mark; stop to save."
  else
    echo "ODOM: map = floor frame at the start mark."
  fi
fi
# viewer / executor clouds; not in pids: a relay crash must not stop SLAM (map_builder publishes the map when mapping)
python3 "$tools/cloud_relay.py" $([[ "$mode" == mapping ]] && echo --no-map) > /data/logs/cloud_relay.log 2>&1 &
relay=$!
wait -n "${pids[@]}"; echo "a SLAM process exited; stopping the others (logs in /data/logs)" >&2
