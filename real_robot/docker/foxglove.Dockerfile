# Foxglove viewer bridge (foxglove_bridge, C++): the robot's Foxy nodes over Cyclone DDS 0.10 (vendor build on the
# robot, Humble's own 0.10 here: same wire protocol, standard message types only). Messages go to the browser as their
# raw CDR bytes (no Python, no JSON); per-client send buffer with drop instead of an ever-growing queue.
# Built on the robot (g1.sh build foxglove); config real_robot/config/foxglove_bridge.yaml mounted by g1.sh viz.
FROM ros:humble-ros-base-jammy
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ros-humble-foxglove-bridge ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 HOME=/tmp/fg_home ROS_LOG_DIR=/tmp/ros_logs
ENTRYPOINT ["bash", "-c", "source /opt/ros/humble/setup.bash && exec ros2 run foxglove_bridge foxglove_bridge --ros-args --params-file /opt/foxglove_bridge.yaml"]
