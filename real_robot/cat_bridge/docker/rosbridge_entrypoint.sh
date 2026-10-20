#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/foxy/setup.bash
export PYTHONPATH="/opt/rosbridge_overlay/python:${PYTHONPATH}"
set -u
# Same arguments as the repo's visualization/rosbridge/run_rosbridge.sh, except the
# bind address (default: robot localhost only; reach it through an SSH tunnel).
exec ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
  port:="${ROSBRIDGE_PORT:-9090}" address:="${ROSBRIDGE_ADDRESS:-127.0.0.1}" \
  'topics_glob:="[*]"' 'services_glob:="[/rosapi/*]"' 'params_glob:="[/__disabled__]"'
