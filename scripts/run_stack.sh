#!/usr/bin/env bash
set -euo pipefail

mode="${1:?mode required}"
map_path="${2:-${G1_SLAM_MAP_PATH:-}}"
fastlio_pid=""
localizer_pid=""
ransac_pid=""

stop_children() {
  trap - INT TERM EXIT
  if [[ "$mode" == mapping ]] && [[ -n "$fastlio_pid" ]] && kill -0 "$fastlio_pid" 2>/dev/null; then
    echo "requesting final map save: $map_path"
    timeout 30 ros2 service call /g1_slam/map_save std_srvs/srv/Trigger '{}' \
      >> /data/logs/map_save.log 2>&1 || \
      echo "map-save service failed; Fast-LIO will retry during shutdown" >&2
  fi
  for pid in "$ransac_pid" "$localizer_pid" "$fastlio_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
  wait || true
}
trap stop_children INT TERM EXIT

mkdir -p /data/logs /data/maps

existing_nodes="$(ros2 node list --no-daemon 2>/dev/null || true)"
if grep -Eq '(^|/)(fast_lio|fastlio_mapping|laserMapping)$|^/g1_slam/' <<<"$existing_nodes"; then
  echo "refusing to start beside another Fast-LIO/G1 SLAM node:" >&2
  grep -E '(^|/)(fast_lio|fastlio_mapping|laserMapping)$|^/g1_slam/' <<<"$existing_nodes" >&2
  exit 6
fi

for _ in $(seq 1 20); do
  lidar_type="$(ros2 topic type /livox/lidar 2>/dev/null || true)"
  imu_type="$(ros2 topic type /livox/imu 2>/dev/null || true)"
  [[ "$lidar_type" == "livox_ros_driver2/msg/CustomMsg" ]] && \
    [[ "$imu_type" == "sensor_msgs/msg/Imu" ]] && break
  sleep 1
done
if [[ "$lidar_type" != "livox_ros_driver2/msg/CustomMsg" ]] || \
   [[ "$imu_type" != "sensor_msgs/msg/Imu" ]]; then
  echo "vendor inputs unavailable or wrong type:" >&2
  echo "  /livox/lidar: ${lidar_type:-not discovered}" >&2
  echo "  /livox/imu:   ${imu_type:-not discovered}" >&2
  exit 7
fi

mapping_args=()
if [[ "$mode" == mapping ]]; then
  run_id="${G1_SLAM_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  map_path="/data/maps/${run_id}.pcd"
  if [[ -e "$map_path" ]]; then
    echo "refusing to overwrite existing map: $map_path" >&2
    exit 3
  fi
  mapping_args=(-p pcd_save.pcd_save_en:=true -p map_file_path:="$map_path")
  echo "mapping output: $map_path"
fi

if [[ "$mode" == localization ]]; then
  if [[ -z "$map_path" || ! -r "$map_path" ]]; then
    echo "localization requires a readable map path under /data/maps" >&2
    exit 4
  fi
fi

ros2 run fast_lio fastlio_mapping --ros-args \
  -r __node:=fast_lio -r __ns:=/g1_slam \
  --params-file /opt/g1_slam/config/mid360_g1.yaml \
  -r /cloud_registered:=/g1_slam/cloud_registered \
  -r /cloud_registered_body:=/g1_slam/cloud_registered_body \
  -r /cloud_effected:=/g1_slam/cloud_effected \
  -r /Laser_map:=/g1_slam/laser_map \
  -r /Odometry:=/g1_slam/odometry \
  -r /path:=/g1_slam/path \
  "${mapping_args[@]}" > /data/logs/fast_lio.log 2>&1 &
fastlio_pid=$!

# Publish an additional, floor-aligned visualization frame/cloud.  This process
# never replaces Fast-LIO's world->camera_init transform or estimator outputs.
python3 /opt/g1_slam/ransac_gravity_override.py --publish \
  --input_topic /g1_slam/cloud_registered \
  --output_topic /g1_viz/cloud_registered_floor \
  --output_frame camera_init_floor \
  --tf_timeout 30 \
  --max_correction_deg 15 \
  --min_inlier_ratio 0.20 \
  > /data/logs/ransac_floor.log 2>&1 &
ransac_pid=$!

if [[ "$mode" == localization ]]; then
  gravity_ready=false
  for _ in $(seq 1 60); do
    kill -0 "$fastlio_pid" 2>/dev/null || {
      echo "Fast-LIO exited during startup; see /data/logs/fast_lio.log" >&2
      exit 5
    }
    if grep -q 'Gravity-aligned world frame published' /data/logs/fast_lio.log 2>/dev/null; then
      gravity_ready=true
      break
    fi
    sleep 1
  done
  if [[ "$gravity_ready" != true ]]; then
    echo "Fast-LIO gravity initialization timed out; see /data/logs/fast_lio.log" >&2
    exit 8
  fi

  # NDT consumes Fast-LIO's motion-corrected body-frame scan, while the
  # absolute Fast-LIO odometry supplies the between-scan prediction. The
  # localizer publishes map->world, completing map->world->camera_init->body.
  ros2 run lidar_localization_ros2 lidar_localization_node --ros-args \
    -r __node:=ndt_localizer -r __ns:=/g1_localization \
    --params-file /opt/g1_slam/config/ndt_localization_g1.yaml \
    -p map_path:="$map_path" \
    -r cloud:=/g1_slam/cloud_registered_body \
    -r odom:=/g1_slam/odometry \
    -r initialpose:=/initialpose \
    > /data/logs/ndt_localizer.log 2>&1 &
  localizer_pid=$!

  change_state_service=/g1_localization/ndt_localizer/change_state
  lifecycle_ready=false
  for _ in $(seq 1 40); do
    kill -0 "$localizer_pid" 2>/dev/null || {
      echo "NDT localizer exited during startup; see /data/logs/ndt_localizer.log" >&2
      exit 9
    }
    if ros2 service list --no-daemon 2>/dev/null | grep -qx "$change_state_service"; then
      lifecycle_ready=true
      break
    fi
    sleep 0.25
  done
  if [[ "$lifecycle_ready" != true ]]; then
    echo "NDT lifecycle service did not appear; see /data/logs/ndt_localizer.log" >&2
    exit 10
  fi

  configure_output="$(timeout 30 ros2 service call "$change_state_service" \
    lifecycle_msgs/srv/ChangeState '{transition: {id: 1}}' 2>&1)" || {
      printf '%s\n' "$configure_output" >&2
      echo "NDT configure transition failed" >&2
      exit 11
    }
  if ! grep -Eq 'success[=:][[:space:]]*(True|true)' <<<"$configure_output"; then
    printf '%s\n' "$configure_output" >&2
    echo "NDT configure transition was rejected" >&2
    exit 11
  fi

  activate_output="$(timeout 30 ros2 service call "$change_state_service" \
    lifecycle_msgs/srv/ChangeState '{transition: {id: 3}}' 2>&1)" || {
      printf '%s\n' "$activate_output" >&2
      echo "NDT activate transition failed" >&2
      exit 12
    }
  if ! grep -Eq 'success[=:][[:space:]]*(True|true)' <<<"$activate_output"; then
    printf '%s\n' "$activate_output" >&2
    echo "NDT activate transition was rejected" >&2
    exit 12
  fi
  echo "NDT localization active: map=$map_path"
fi

wait "$fastlio_pid"
