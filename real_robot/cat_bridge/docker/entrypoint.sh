#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/foxy/setup.bash
source /opt/g1_cat_bridge/setup.bash
set -u
case "${1:-run}" in
  run)
    ros2 run g1_cat_bridge utlidar_to_livox & conv=$!
    ros2 run g1_cat_bridge stationary_pose & pose=$!
    trap 'kill -INT $conv $pose 2>/dev/null; wait' INT TERM
    wait -n $conv $pose; rc=$?
    echo "a bridge process exited (rc=$rc); stopping the other" >&2
    kill -INT $conv $pose 2>/dev/null || true; wait || true; exit "$rc" ;;
  converter) exec ros2 run g1_cat_bridge utlidar_to_livox ;;
  pose) exec ros2 run g1_cat_bridge stationary_pose ;;
  *) echo "usage: run|converter|pose" >&2; exit 2 ;;
esac
