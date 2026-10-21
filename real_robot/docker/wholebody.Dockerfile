# Whole-body policy executor (real_robot/wholebody) + its unitree_hg messages (incl. LowCmd).
# Build context: repository root. Targets: runtime (robot), dev (laptop: + MuJoCo for the simulated G1).
FROM ros:foxy-ros-base-focal AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends cmake g++ gfortran python3-colcon-common-extensions \
    python3-dev python3-pip python3-wheel ros-foxy-rosidl-default-generators && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir "numpy==1.24.4" && \
    python3 -m pip wheel --no-deps --no-build-isolation "scikit-fmm==2023.4.2" --wheel-dir /wheels
COPY real_robot/ros/unitree_hg /ws/src/unitree_hg
RUN . /opt/ros/foxy/setup.sh && cd /ws && colcon build --merge-install --cmake-args -DBUILD_TESTING=OFF

FROM ros:foxy-ros-base-focal AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip ros-foxy-rmw-cyclonedds-cpp \
    ros-foxy-tf2-ros ros-foxy-visualization-msgs ros-foxy-rosidl-default-runtime && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /tmp/wheels
# numpy/scipy from pip (focal's 1.17/1.3 are too old for ONNX Runtime); these shadow the apt ones for this image only
RUN python3 -m pip install --no-cache-dir "numpy==1.24.4" "scipy==1.10.1" "onnxruntime==1.16.3" /tmp/wheels/*.whl && rm -rf /tmp/wheels
COPY vendor/unitree_cyclone /tmp/unitree_cyclone
RUN if [ "$(uname -m)" = aarch64 ]; then \
      test -f /tmp/unitree_cyclone/libddsc.so || { echo "run scripts/fetch_runtime.sh first" >&2; exit 1; }; \
      cp /tmp/unitree_cyclone/libddsc.so /usr/local/lib/libddsc.so && \
      cp /tmp/unitree_cyclone/librmw_cyclonedds_cpp.so /opt/ros/foxy/lib/librmw_cyclonedds_cpp.so && \
      ln -sf libddsc.so /usr/local/lib/libddsc.so.0 && ldconfig; fi; rm -rf /tmp/unitree_cyclone
COPY --from=builder /ws/install /opt/wb/messages
COPY cat/state_core.py cat/perception_core.py cat/wholebody_core.py cat/wholebody_fields.py cat/wholebody_sim.py /opt/wb/cat/
COPY cat/wholebody /opt/wb/cat/wholebody
COPY cat/tests/test_wholebody.py cat/tests/wholebody_reference.npz /opt/wb/cat/tests/
COPY real_robot/wholebody/g1_wholebody /opt/wb/g1_wholebody
COPY real_robot/wholebody/tests /opt/wb/tests
COPY third_party/unitree_sdk2_python/unitree_sdk2py/utils/lib/crc_amd64.so third_party/unitree_sdk2_python/unitree_sdk2py/utils/lib/crc_aarch64.so /opt/wb/sdk_crc/
COPY real_robot/docker/wholebody_entrypoint.sh /opt/wb/entrypoint.sh
RUN chmod 0555 /opt/wb/entrypoint.sh && mkdir -p /data/logs && chown -R 1000:1000 /data
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 HOME=/tmp/wb_home ROS_LOG_DIR=/tmp/ros_logs \
    LD_LIBRARY_PATH=/usr/local/lib:/opt/ros/foxy/lib/aarch64-linux-gnu:/opt/ros/foxy/lib \
    PYTHONPATH=/opt/wb:/opt/wb/cat G1_SDK_CRC_DIR=/opt/wb/sdk_crc \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 1000:1000
STOPSIGNAL SIGINT
LABEL com.g1-real-robot.role=wholebody
ENTRYPOINT ["/opt/wb/entrypoint.sh"]
CMD ["validate"]

FROM runtime AS dev
USER root
RUN python3 -m pip install --no-cache-dir "mujoco==3.1.6" && apt-get update && \
    apt-get install -y --no-install-recommends libegl1 libgl1 ffmpeg && rm -rf /var/lib/apt/lists/*
USER 1000:1000
