# SLAM for the whole-body pipeline: g1_sensor_restamp + FAST_LIO_G1 + NDT localisation (no Nav2, no Open3D).
# Build context: repository root.  arm64 (robot) uses the repo's Unitree CycloneDDS runtime; amd64 (laptop) the stock one.
FROM ros:foxy-ros-base-focal AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ccache cmake libeigen3-dev libpcl-dev zlib1g-dev python3-colcon-common-extensions python3-dev \
    ros-foxy-diagnostic-msgs ros-foxy-lifecycle-msgs ros-foxy-pcl-conversions ros-foxy-pcl-ros \
    ros-foxy-rclcpp-components ros-foxy-rclcpp-lifecycle ros-foxy-std-srvs ros-foxy-tf2-eigen \
    ros-foxy-tf2-geometry-msgs ros-foxy-tf2-ros ros-foxy-tf2-sensor-msgs ros-foxy-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/g1_slam_ws
# One layer per package, most stable first (FAST_LIO_G1 last: it is the one being updated): a change in one package rebuilds only that package
# and the ones after it (lidar_localization_ros2 alone takes ~40 min under QEMU). ccache in a BuildKit cache mount
# keeps object files across builds, so even a rebuilt package recompiles only the files that changed.
# Legacy (non-BuildKit) builders: real_robot/scripts/g1.sh strips the cache mounts.
# Robot defaults; the laptop's arm64 (QEMU) build raises MAKE_JOBS.
ARG MAKE_JOBS=4
ARG COLCON_WORKERS=2
ENV CCACHE_DIR=/ccache CCACHE_MAXSIZE=5G
ARG COLCON="colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache --packages-select"
COPY src/livox_ros_driver2 src/livox_ros_driver2
RUN --mount=type=cache,target=/ccache . /opt/ros/foxy/setup.sh && MAKEFLAGS=-j${MAKE_JOBS} ${COLCON} livox_ros_driver2
COPY src/ndt_omp_ros2 src/ndt_omp_ros2
RUN --mount=type=cache,target=/ccache . /opt/ros/foxy/setup.sh && MAKEFLAGS=-j${MAKE_JOBS} ${COLCON} ndt_omp_ros2
COPY src/lidar_localization_ros2 src/lidar_localization_ros2
RUN --mount=type=cache,target=/ccache . /opt/ros/foxy/setup.sh && MAKEFLAGS=-j${MAKE_JOBS} ${COLCON} lidar_localization_ros2
COPY src/g1_sensor_restamp src/g1_sensor_restamp
RUN --mount=type=cache,target=/ccache . /opt/ros/foxy/setup.sh && MAKEFLAGS=-j${MAKE_JOBS} ${COLCON} g1_sensor_restamp
COPY src/fast_lio src/fast_lio
RUN --mount=type=cache,target=/ccache . /opt/ros/foxy/setup.sh && MAKEFLAGS=-j${MAKE_JOBS} ${COLCON} fast_lio

FROM ros:foxy-ros-base-focal AS runtime
ENV DEBIAN_FRONTEND=noninteractive
# Only the PCL runtime libraries the binaries link (checked with ldd). ros-foxy-pcl-ros / pcl-conversions are not
# linked but depend on libpcl-dev (all PCL/VTK/Qt headers): without them the image is ~2 GB smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libpcl-common1.10 libpcl-features1.10 libpcl-filters1.10 libpcl-io1.10 libpcl-kdtree1.10 \
    libpcl-ml1.10 libpcl-octree1.10 libpcl-registration1.10 libpcl-sample-consensus1.10 libpcl-search1.10 \
    libpcl-segmentation1.10 libpcl-visualization1.10 python3-numpy python3-yaml \
    ros-foxy-diagnostic-msgs ros-foxy-lifecycle-msgs ros-foxy-rclcpp-lifecycle ros-foxy-rmw-cyclonedds-cpp \
    ros-foxy-std-srvs ros-foxy-tf2-eigen ros-foxy-tf2-geometry-msgs ros-foxy-tf2-ros ros-foxy-tf2-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*
COPY vendor/unitree_cyclone /tmp/unitree_cyclone
COPY vendor/licenses /usr/share/doc/g1-slam-runtime-licenses
RUN if [ "$(uname -m)" = aarch64 ]; then \
      test -f /tmp/unitree_cyclone/libddsc.so || { echo "run scripts/fetch_runtime.sh first" >&2; exit 1; }; \
      cp /tmp/unitree_cyclone/libddsc.so /usr/local/lib/libddsc.so && \
      cp /tmp/unitree_cyclone/librmw_cyclonedds_cpp.so /opt/ros/foxy/lib/librmw_cyclonedds_cpp.so && \
      ln -sf libddsc.so /usr/local/lib/libddsc.so.0 && ldconfig; fi; rm -rf /tmp/unitree_cyclone
COPY --from=builder /opt/g1_slam_ws/install /opt/g1_slam_ws/install
COPY real_robot/config /opt/real_robot/config
COPY real_robot/slam /opt/real_robot/slam
RUN chmod 0555 /opt/real_robot/slam/*.sh /opt/real_robot/slam/*.py && mkdir -p /data/maps /data/logs && chown -R 1000:1000 /data
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 HOME=/tmp/g1_slam_home ROS_LOG_DIR=/data/logs/ros \
    LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/foxy/lib/aarch64-linux-gnu:/opt/ros/foxy/lib \
    PYTHONPATH=/opt/real_robot/slam OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
USER 1000:1000
LABEL com.g1-real-robot.role=slam
ENTRYPOINT ["/opt/real_robot/slam/run_slam.sh"]
CMD ["odom"]

# Laptop only: bag replay (laptop.sh map/localize/lio). The robot image is --target runtime.
FROM runtime AS dev
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ros-foxy-ros2bag ros-foxy-rosbag2-storage-default-plugins \
    && rm -rf /var/lib/apt/lists/*
USER 1000:1000
