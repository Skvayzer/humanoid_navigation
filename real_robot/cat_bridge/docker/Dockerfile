# Read-only sensor adapter + stationary pose for running CAT perception on a G1
# whose MID360 is published by the vendor main board. Build context is staged by
# scripts/perception.sh (bridge package, repo's livox msgs, repo's Unitree Cyclone libs).
FROM ros:foxy-ros-base-focal AS builder
COPY livox_ros_driver2 /ws/src/livox_ros_driver2
COPY g1_cat_bridge /ws/src/g1_cat_bridge
WORKDIR /ws
RUN . /opt/ros/foxy/setup.sh && colcon build --merge-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

FROM ros:foxy-ros-base-focal
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-foxy-rmw-cyclonedds-cpp python3-numpy && rm -rf /var/lib/apt/lists/*
# Same vendor-compatible DDS runtime as the repo's CAT/SLAM images.
COPY unitree_cyclone/libddsc.so /usr/local/lib/libddsc.so
COPY unitree_cyclone/librmw_cyclonedds_cpp.so /opt/ros/foxy/lib/librmw_cyclonedds_cpp.so
RUN ln -s libddsc.so /usr/local/lib/libddsc.so.0 && ldconfig
COPY --from=builder /ws/install /opt/g1_cat_bridge
COPY entrypoint.sh /opt/g1_cat_bridge/entrypoint.sh
RUN chmod 0555 /opt/g1_cat_bridge/entrypoint.sh
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/foxy/lib/aarch64-linux-gnu:/opt/ros/foxy/lib \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOG_DIR=/tmp/ros_logs \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
USER 1000:1000
STOPSIGNAL SIGINT
LABEL com.g1-cat-bridge.role=read-only-sensor-adapter
ENTRYPOINT ["/opt/g1_cat_bridge/entrypoint.sh"]
CMD ["run"]
