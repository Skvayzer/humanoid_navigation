# Install on another robot

This is a snapshot of a working **Unitree G1 / Ubuntu 20.04 / ARM64 / ROS 2 Foxy**
deployment. It is not a generic safe-to-run humanoid image. An X2, another sensor
mount, a different Unitree firmware, or an x86 PC needs adaptation and validation.
Nothing below puts the robot into locomotion mode automatically.

## 1. Inspect before installing

Keep the robot disarmed. Record existing containers, ROS installations, sensor
drivers and startup services. Do not uninstall packages, overwrite a vendor
workspace, change the default ROS environment, or stop an unrelated project.

Required: Docker, Git, curl, Bash, Python 3; host Foxy and the robot's working
CycloneDDS + Livox overlays for the optional host components. The existing driver
must publish `/livox/lidar` (`livox_ros_driver2/msg/CustomMsg`) and `/livox/imu`
(`sensor_msgs/msg/Imu`). This repository supplies the message interface, **not a
second hardware driver**. Match the wire definition to your driver.

The original computer has 16 GB RAM. Container defaults are 6 CPU cores and a
10 GB memory ceiling. Review available resources before building or running.

## 2. Clone into a new directory

```bash
git clone https://github.com/Skvayzer/humanoid_navigation.git
cd humanoid_navigation
cp config/site.env.example config/site.env
./scripts/verify_source.sh
./scripts/fetch_runtime.sh
```

The dependency download is checksum-verified. It contains the exact deployed
ARM64 Open3D CPU extension and two CycloneDDS shared libraries, **not a Docker
image or a map**. GUI assets from Open3D are unnecessary with the deployed
headless importer and are omitted. Matching middleware source is a separate
release asset; see [provenance](../PROVENANCE.md).

Edit `config/site.env` for workspace/runtime paths. Inspect
`config/cyclonedds.xml` for the container's DDS interface (`eth0` by default).
Host visualization and the gateway use `G1_HOST_CYCLONEDDS_URI`; the SDK worker
uses `G1_SDK_INTERFACE` on its separate DDS participant. ROS domain defaults to
0. These are robot-internal network choices, not necessarily the Wi-Fi interface.

## 3. Install shortcuts without replacing existing ones

```bash
./scripts/install_shortcuts.sh
export PATH="$HOME/bin:$PATH"
```

The installer refuses existing commands from another project. For a side-by-side
checkout use `G1_BIN_DIR="$HOME/humanoid_navigation_bin" ./scripts/install_shortcuts.sh`
and explicitly use that directory in your shell's PATH. It does not edit your
shell startup files. A second checkout is **not** permission to launch another
SLAM publisher in the same ROS domain: node names and TF frames are shared.

## 4. Build and validate offline

```bash
./scripts/build_image.sh
./scripts/run_container.sh validate
```

The image contains Fast-LIO, NDT-OMP, the NDT localizer, floor visualization,
Nav2 Navfn, Regulated Pure Pursuit and their launch code. No Humble installation
is used. Docker installs dependencies inside the image, not into host ROS.

The runtime has a read-only root, dropped capabilities, no privileged mode and
no host device mounts. Only project config/nav files (read-only) and dedicated
maps/logs (read-write) are mounted. Host networking intentionally discovers the
vendor topics; it is **not DDS isolation or a robot safety boundary**. CPU, RAM,
network bandwidth and TF/topic names remain shared resources.

Foxy is end-of-life. Sources and native runtime hashes are captured; Ubuntu/ROS
apt dependencies still come from the configured repositories. This is a
rebuildable source package, not a promise of a byte-identical/offline image.
The manual GitHub Actions ARM64 build tests compilation and offline imports.

## 5. Bring your map and calibration

Public Git and releases deliberately exclude lab maps. Copy your private aligned
PCD and its matching edited occupancy grid to `$G1_SLAM_RUNTIME_DIR/maps` (default
`$HOME/g1_slam_runtime/maps`). The existing lab configuration expects:

```text
maps/
  teaching_lab_aligned.pcd
  teaching_lab_nav/
    teaching_lab_nav.yaml
    teaching_lab_nav.pgm
```

The YAML must retain the matching resolution/origin; its image path must resolve
inside the container. Set `G1_SLAM_DEFAULT_MAP` and `G1_NAV_MAP_YAML` in site.env
for other filenames. The latter is a **container** path beneath `/data/maps`.

Do not reuse the lab's NDT initial pose on a different map. Review
`config/ndt_localization_g1.yaml`, the LiDAR/IMU transform in `mid360_g1.yaml`, the
planar base frame, and Nav2 footprint. See [maps and frames](MAPS_AND_FRAMES.md).
Keep the robot motionless during Fast-LIO initialization. Confirm map/scan/pose
agreement before sending any navigation goal.

## 6. Optional host visualization

Install only the named Foxy visualization packages after reviewing apt's proposed
changes; do not run a distribution upgrade or autoremove:

```bash
sudo apt update
sudo apt install ros-foxy-rosbridge-server ros-foxy-rosapi
foxglove-start
```

The deployed baseline was rosbridge/rosapi **1.3.1**. Our copied Python overlay
patches only this launcher through PYTHONPATH, leaving `/opt/ros` unchanged. Do not
assume the patches suit a different rosbridge version.

Use **Rosbridge**, not Foxglove WebSocket, in Foxglove:
`ws://<robot-ip>:9090`. The deployed client was Foxglove 3.0.0. Topic publication
through this bridge is limited to `/g1_nav/goal_pose`, service access to `/rosapi/*`.
It is not an authenticated Internet-facing service. Use a trusted robot network
or SSH tunnel; do not expose port 9090 publicly.

For raw Livox points in a standard PointCloud2:

```bash
source scripts/env.sh
source /opt/ros/foxy/setup.bash
source "$G1_CYCLONE_WS/install/setup.bash"
source "$G1_LIVOX_WS/install/setup.bash"
cd visualization/livox
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
./run_livox_viz.sh
```

Do not run this adapter alongside an already-running copy. It publishes
`/g1_viz/livox_points` and IMU markers; vendor sensor input is not altered.

## 7. Optional host motion gateway — last, not first

The gateway runs outside Docker, as on the deployed G1. The exact Unitree SDK
source is vendored in `third_party/unitree_sdk2_python`. Existing working host
dependencies can be retained. For a new installation, use a dedicated Python 3.8
venv with access to host ROS/NumPy, not a global pip upgrade:

```bash
sudo apt install python3-venv python3-dev python3-numpy
python3 -m venv --system-site-packages "$HOME/.venvs/g1-motion"
source scripts/env.sh
source /opt/ros/foxy/setup.bash
source "$G1_CYCLONE_WS/install/setup.bash"
# Point this at the prefix containing lib/libddsc.so and include/dds if needed.
# CYCLONEDDS_HOME="$G1_CYCLONE_WS/install/cyclonedds" \
"$HOME/.venvs/g1-motion/bin/pip" install -r host/motion-requirements.txt
```

Set `G1_MOTION_PYTHON="$HOME/.venvs/g1-motion/bin/python3"` in site.env. The SDK
source is imported from the checkout; its unrelated examples and optional camera
dependencies are not needed by this gateway. Check that `cyclonedds==0.10.2` and
`rclpy` import in the selected environment before launch.

If the robot lacks compatible host DDS/Livox overlays, that is a prerequisite to
resolve in a **new user-owned workspace**, not by replacing the vendor install.
The archived CycloneDDS/RMW source is supplied for this purpose. Host middleware
building, firmware compatibility and actual walking need target-specific tests.

Follow [COMMANDS.md](COMMANDS.md) for preview and supervised motion. Never run SDK
example programs to test installation: some change modes or issue motor commands.
