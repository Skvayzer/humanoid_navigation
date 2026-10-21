# Demo recorder (real_robot/demo/ar_demo.py): head camera via videohub + AR overlay of the virtual world, rosbag of the
# viewer topics. Built on the robot (g1.sh build demo); the script itself is mounted from the checkout.
FROM ros:foxy-ros-base-focal AS builder
RUN apt-get update && apt-get install -y --no-install-recommends python3-colcon-common-extensions && rm -rf /var/lib/apt/lists/*
COPY real_robot/ros/unitree_api /ws/src/unitree_api
RUN . /opt/ros/foxy/setup.sh && cd /ws && colcon build --merge-install --cmake-args -DBUILD_TESTING=OFF

FROM ros:foxy-ros-base-focal AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3-opencv python3-numpy ros-foxy-rmw-cyclonedds-cpp \
    ros-foxy-tf2-ros ros-foxy-visualization-msgs ros-foxy-ros2bag ros-foxy-rosbag2-transport ros-foxy-rosbag2-storage-default-plugins \
    && rm -rf /var/lib/apt/lists/*
COPY vendor/unitree_cyclone /tmp/unitree_cyclone
RUN if [ "$(uname -m)" = aarch64 ]; then \
      cp /tmp/unitree_cyclone/libddsc.so /usr/local/lib/libddsc.so && \
      cp /tmp/unitree_cyclone/librmw_cyclonedds_cpp.so /opt/ros/foxy/lib/librmw_cyclonedds_cpp.so && \
      ln -sf libddsc.so /usr/local/lib/libddsc.so.0 && ldconfig; fi; rm -rf /tmp/unitree_cyclone
COPY --from=builder /ws/install /opt/demo/messages
# the vendor CycloneDDS 0.10 first on the library path: Foxy's 0.7 rejects the robot's DDS config
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 HOME=/tmp/demo_home ROS_LOG_DIR=/tmp/ros_logs PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/foxy/lib/aarch64-linux-gnu:/opt/ros/foxy/lib
LABEL com.g1-real-robot.role=demo
ENTRYPOINT ["bash", "-c", "source /opt/ros/foxy/setup.bash && source /opt/demo/messages/setup.bash && exec python3 /opt/demo/ar_demo.py \"$@\"", "--"]
