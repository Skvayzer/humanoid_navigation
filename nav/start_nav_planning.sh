#!/usr/bin/env bash
set -eo pipefail

pid_file=/data/logs/nav/planning_supervisor.pid
log_file=/data/logs/nav/planning.log
mkdir -p /data/logs/nav

if [[ -r "$pid_file" ]]; then
  old_pid="$(cat "$pid_file")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "planning-only navigation is already running (pid $old_pid)" >&2
    exit 3
  fi
fi

source /opt/ros/foxy/setup.bash
source /opt/g1_slam_ws/install/setup.bash
set -u
export ROS_LOG_DIR=/data/logs/ros
export G1_NAV_IGNORE_GOAL_YAW=false
# Final-heading Spin uses the recovery server's separate velocity output. Keep
# it disabled until that path is deliberately integrated with the safety gate.
# RPP still turns toward the path before commanding forward motion.
export G1_NAV_FINAL_SPIN_TO_GOAL_YAW=false
export G1_NAV_FINAL_YAW_TOLERANCE_RAD=0.10
export G1_NAV_POSE_HZ=10.0
echo "$$" > "$pid_file"
launch_pid=""

cleanup() {
  trap - INT TERM EXIT
  if [[ -n "$launch_pid" ]] && kill -0 "$launch_pid" 2>/dev/null; then
    kill -TERM "$launch_pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}
trap cleanup INT TERM EXIT

echo "SAFETY: X2-style Foxy Nav2 with live local costmap; cmd_vel quarantined behind the separately armed gateway" >> "$log_file"
nice -n 10 ros2 launch /opt/g1_slam/nav/g1_nav_planning.launch.py >>"$log_file" 2>&1 &
launch_pid=$!
wait "$launch_pid"
