FROM ros:foxy-ros-base-focal AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ liboctomap-dev python3-colcon-common-extensions python3-dev \
    python3-pip python3-wheel python3-numpy \
    ros-foxy-rosidl-default-generators && rm -rf /var/lib/apt/lists/*
COPY cat/native /build/src
RUN cmake -S /build/src -B /build/out -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /build/out --parallel 2
COPY src/livox_ros_driver2 /build/messages/src/livox_ros_driver2
COPY cat/interfaces/unitree_hg /build/messages/src/unitree_hg
WORKDIR /build/messages
RUN . /opt/ros/foxy/setup.sh && colcon build --merge-install --cmake-args -DBUILD_TESTING=OFF
RUN python3 -m pip wheel --no-deps --no-build-isolation \
    scikit-fmm==2023.4.2 --wheel-dir /build/wheels

FROM ros:foxy-ros-base-focal
RUN apt-get update && apt-get install -y --no-install-recommends \
    liboctomap1.9 python3-numpy python3-scipy python3-pip \
    ros-foxy-rmw-cyclonedds-cpp ros-foxy-rosidl-default-runtime \
    ros-foxy-tf2-ros ros-foxy-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/wheels /tmp/cat-wheels
RUN python3 -m pip install --no-deps --no-index /tmp/cat-wheels/*.whl && rm -r /tmp/cat-wheels
# Match the already-tested G1 DDS runtime, without changing the host libraries.
COPY vendor/unitree_cyclone/libddsc.so /usr/local/lib/libddsc.so
COPY vendor/unitree_cyclone/librmw_cyclonedds_cpp.so /opt/ros/foxy/lib/librmw_cyclonedds_cpp.so
COPY vendor/licenses /usr/share/doc/g1-cat-runtime-licenses
RUN ln -s libddsc.so /usr/local/lib/libddsc.so.0 && ldconfig
COPY cat/perception_core.py cat/preview_node.py /opt/cat/
COPY cat/field_core.py cat/state_core.py cat/research_node.py cat/checkpoint.py cat/checkpoint_manifest.json /opt/cat/
COPY cat/interfaces/unitree_hg/LICENSE /usr/share/doc/g1-cat-runtime-licenses/unitree_hg-LICENSE
COPY cat/config /opt/cat/config
COPY cat/tests /opt/cat/tests
COPY cat/upstream /opt/cat/upstream
COPY --from=builder /build/out/libcat_octomap.so /opt/cat/build/libcat_octomap.so
COPY --from=builder /build/messages/install /opt/cat/messages
COPY cat/scripts/entrypoint.sh /opt/cat/entrypoint.sh
RUN chmod 0555 /opt/cat/entrypoint.sh
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/foxy/lib/aarch64-linux-gnu:/opt/ros/foxy/lib \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 \
    ROS_LOG_DIR=/tmp/ros_logs PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 1000:1000
STOPSIGNAL SIGINT
WORKDIR /opt/cat
LABEL com.g1-cat.role=perception-only com.g1-cat.upstream=866ba392f1c1e84b92ad75fa66550f26e8af8e48
ENTRYPOINT ["/opt/cat/entrypoint.sh"]
CMD ["validate"]
