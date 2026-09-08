FROM ros:foxy-ros-base-focal AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libeigen3-dev \
    libpcl-dev \
    python3-colcon-common-extensions \
    python3-dev \
    ros-foxy-diagnostic-msgs \
    ros-foxy-lifecycle-msgs \
    ros-foxy-pcl-conversions \
    ros-foxy-pcl-ros \
    ros-foxy-rclcpp-components \
    ros-foxy-rclcpp-lifecycle \
    ros-foxy-rmw-cyclonedds-cpp \
    ros-foxy-std-srvs \
    ros-foxy-tf2-eigen \
    ros-foxy-tf2-geometry-msgs \
    ros-foxy-tf2-ros \
    ros-foxy-tf2-sensor-msgs \
    ros-foxy-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/g1_slam_ws
COPY src/livox_ros_driver2 src/livox_ros_driver2
COPY src/fast_lio src/fast_lio
COPY src/ndt_omp_ros2 src/ndt_omp_ros2
COPY src/lidar_localization_ros2 src/lidar_localization_ros2
RUN . /opt/ros/foxy/setup.sh && \
    MAKEFLAGS=-j2 colcon build --merge-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

FROM ros:foxy-ros-base-focal AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libc++1 \
    libc++abi1 \
    libgl1 \
    libgomp1 \
    libpcl-common1.10 \
    libpcl-io1.10 \
    libpcl-kdtree1.10 \
    libpcl-search1.10 \
    libpcl-filters1.10 \
    libpcl-registration1.10 \
    libx11-6 \
    python3-numpy \
    python3-scipy \
    ros-foxy-diagnostic-msgs \
    ros-foxy-lifecycle-msgs \
    ros-foxy-nav2-bringup \
    ros-foxy-nav2-regulated-pure-pursuit-controller \
    ros-foxy-navigation2 \
    ros-foxy-pcl-conversions \
    ros-foxy-pcl-ros \
    ros-foxy-rclcpp-lifecycle \
    ros-foxy-rmw-cyclonedds-cpp \
    ros-foxy-std-srvs \
    ros-foxy-tf2-eigen \
    ros-foxy-tf2-geometry-msgs \
    ros-foxy-tf2-ros \
    ros-foxy-tf2-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*

# Match the custom Cyclone DDS runtime used by the G1 Livox publisher. The
# stock Foxy Cyclone participant crashes against the vendor DDS environment.
COPY vendor/unitree_cyclone/libddsc.so /usr/local/lib/libddsc.so
COPY vendor/licenses /usr/share/doc/g1-slam-runtime-licenses
COPY vendor/unitree_cyclone/librmw_cyclonedds_cpp.so /opt/ros/foxy/lib/librmw_cyclonedds_cpp.so
RUN ln -sf libddsc.so /usr/local/lib/libddsc.so.0 && ldconfig

COPY --from=builder /opt/g1_slam_ws/install /opt/g1_slam_ws/install
COPY vendor/open3d /usr/local/lib/python3.8/dist-packages/open3d
COPY docker/open3d_headless_init.py /usr/local/lib/python3.8/dist-packages/open3d/__init__.py
COPY src/localizer.py /opt/g1_slam/localizer.py
COPY src/ransac_gravity_override.py /opt/g1_slam/ransac_gravity_override.py
COPY config /opt/g1_slam/config
COPY scripts/container_entrypoint.sh /opt/g1_slam/container_entrypoint.sh
COPY scripts/run_stack.sh /opt/g1_slam/run_stack.sh
COPY nav /opt/g1_slam/nav

RUN chmod 0555 /opt/g1_slam/container_entrypoint.sh \
    /opt/g1_slam/run_stack.sh /opt/g1_slam/localizer.py \
    /opt/g1_slam/ransac_gravity_override.py \
    /opt/g1_slam/nav/g1_nav_planning.launch.py \
    /opt/g1_slam/nav/g1_nav_pose_bridge.py \
    /opt/g1_slam/nav/g1_goal_to_nav.py \
    /opt/g1_slam/nav/g1_path_frame_relay.py \
    /opt/g1_slam/nav/start_nav_planning.sh \
    /opt/g1_slam/nav/stop_nav.sh && \
    mkdir -p /data/maps /data/logs/ros && chown -R 1000:1000 /data

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ROS_DOMAIN_ID=0 \
    LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/foxy/lib/aarch64-linux-gnu:/opt/ros/foxy/lib \
    HOME=/tmp/g1_slam_home \
    ROS_LOG_DIR=/data/logs/ros \
    OPENBLAS_NUM_THREADS=1 \
    OPENBLAS64_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

USER 1000:1000
ENTRYPOINT ["/opt/g1_slam/container_entrypoint.sh"]
CMD ["validate"]

LABEL com.g1-slam.middleware=unitree-cyclonedds
