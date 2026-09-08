#!/usr/bin/env bash
set -u

pid_file=/data/logs/nav/planning_supervisor.pid
if [[ -r "$pid_file" ]]; then
  pid="$(cat "$pid_file")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
fi

for executable in \
  planner_server \
  controller_server \
  recoveries_server \
  bt_navigator \
  map_server \
  lifecycle_manager; do
  pkill -TERM -x "$executable" 2>/dev/null || true
done

# ros2 launch can leave Python or controller children behind if its supervisor
# is stopped during startup. Match only this project's navigation processes.
for pattern in \
  '/opt/g1_slam/nav/[g]1_nav_planning.launch.py' \
  '/opt/g1_slam/nav/[g]1_nav_pose_bridge.py' \
  '/opt/g1_slam/nav/[g]1_lidar_costmap_relay.py' \
  '/opt/g1_slam/nav/[g]1_goal_to_plan.py' \
  '/opt/g1_slam/nav/[g]1_goal_to_nav.py' \
  '/opt/g1_slam/nav/[g]1_path_frame_relay.py' \
  '/opt/ros/foxy/lib/nav2_controller/[c]ontroller_server' \
  '/opt/ros/foxy/lib/nav2_recoveries/[r]ecoveries_server' \
  '/opt/ros/foxy/lib/nav2_bt_navigator/[b]t_navigator' \
  '/opt/ros/foxy/lib/nav2_lifecycle_manager/[l]ifecycle_manager'; do
  pkill -TERM -f "$pattern" 2>/dev/null || true
done

rm -f "$pid_file"
echo "navigation stop requested (SLAM/localization left running)"
