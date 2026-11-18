#!/usr/bin/env bash
# g1-rr-foxglove: domain_bridge (robot domain 0 <-> viewer domain 7, config/viz_domain_bridge.yaml) + foxglove_bridge in
# domain 7. Either one exiting stops the container.
set -eo pipefail
source /opt/ros/humble/setup.bash
ros2 run domain_bridge domain_bridge /opt/viz_domain_bridge.yaml &
ROS_DOMAIN_ID=7 ros2 run foxglove_bridge foxglove_bridge --ros-args --params-file /opt/foxglove_bridge.yaml &
wait -n; kill 0
