# rosbridge (Foxglove "Rosbridge" connection) for viewing /g1_cat/* from a laptop.
# Uses the repo's Foxy compatibility overlay and its Unitree Cyclone runtime.
FROM ros:foxy-ros-base-focal
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-foxy-rosbridge-server ros-foxy-rosapi ros-foxy-rmw-cyclonedds-cpp \
      ros-foxy-visualization-msgs && rm -rf /var/lib/apt/lists/*
COPY unitree_cyclone/libddsc.so /usr/local/lib/libddsc.so
COPY unitree_cyclone/librmw_cyclonedds_cpp.so /opt/ros/foxy/lib/librmw_cyclonedds_cpp.so
RUN ln -s libddsc.so /usr/local/lib/libddsc.so.0 && ldconfig
COPY rosbridge_overlay /opt/rosbridge_overlay/python
COPY rosbridge_entrypoint.sh /opt/rosbridge_entrypoint.sh
RUN chmod 0555 /opt/rosbridge_entrypoint.sh
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/foxy/lib/aarch64-linux-gnu:/opt/ros/foxy/lib \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOG_DIR=/tmp/ros_logs \
    ROS_HOME=/tmp/ros PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/rosbridge_overlay/python
USER 1000:1000
STOPSIGNAL SIGINT
LABEL com.g1-cat-bridge.role=rosbridge-viz
ENTRYPOINT ["/opt/rosbridge_entrypoint.sh"]
