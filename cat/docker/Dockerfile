FROM ros:foxy-ros-base-focal AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ liboctomap-dev && rm -rf /var/lib/apt/lists/*
COPY cat/native /build/src
RUN cmake -S /build/src -B /build/out -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /build/out --parallel 2

FROM ros:foxy-ros-base-focal
RUN apt-get update && apt-get install -y --no-install-recommends \
    liboctomap1.9 python3-numpy python3-scipy \
    ros-foxy-rmw-cyclonedds-cpp ros-foxy-tf2-ros ros-foxy-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*
# Match the already-tested G1 DDS runtime, without changing the host libraries.
COPY vendor/unitree_cyclone/libddsc.so /usr/local/lib/libddsc.so
COPY vendor/unitree_cyclone/librmw_cyclonedds_cpp.so /opt/ros/foxy/lib/librmw_cyclonedds_cpp.so
RUN ln -s libddsc.so /usr/local/lib/libddsc.so.0 && ldconfig
COPY cat/core.py cat/preview_node.py /opt/cat/
COPY cat/config /opt/cat/config
COPY cat/tests /opt/cat/tests
COPY cat/upstream /opt/cat/upstream
COPY --from=builder /build/out/libcat_octomap.so /opt/cat/build/libcat_octomap.so
COPY cat/scripts/entrypoint.sh /opt/cat/entrypoint.sh
RUN chmod 0555 /opt/cat/entrypoint.sh
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/foxy/lib/aarch64-linux-gnu:/opt/ros/foxy/lib \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 \
    ROS_LOG_DIR=/tmp/ros_logs PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 1000:1000
WORKDIR /opt/cat
LABEL com.g1-cat.role=perception-only com.g1-cat.upstream=866ba392f1c1e84b92ad75fa66550f26e8af8e48
ENTRYPOINT ["/opt/cat/entrypoint.sh"]
CMD ["validate"]
