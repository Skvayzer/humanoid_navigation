# FAST_LIO for the Unitree G1

Fork of [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO), branch `ROS2` at `a4743b0`, adapted to the Unitree G1 (Livox Mid-360 in the head) on ROS 2 Foxy. The upstream README, papers and media are on the upstream repository.

Papers: [FAST-LIO2: Fast Direct LiDAR-inertial Odometry](https://arxiv.org/abs/2107.06829) (T-RO 2022), [FAST-LIO](https://arxiv.org/abs/2010.08196) (RA-L 2021).

## Build

Needs ROS 2 (Foxy or newer), PCL, Eigen, and the `livox_ros_driver2` messages (a message-only package is in the same workspace).

```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## Configs

| File | Input |
|---|---|
| `config/g1_mid360.yaml` | G1 vendor stream: `/utlidar/cloud_livox_mid360` (PointCloud2 with per-point `time`) and the restamped `/g1/imu` |
| `config/mid360.yaml` | Upstream defaults for `livox_ros_driver2` `CustomMsg` (`/livox/lidar`, `/livox/imu`) |

## Run

```bash
ros2 launch fast_lio mapping.launch.py                         # config:=<file> to change it
```

On the G1, use `tools/start_fastlio.sh` from the robot workspace instead: it renames the outputs to `/rv/*`.

Outputs: `/Odometry` (IMU pose in `camera_init`), `/path`, `/cloud_registered`, TF `camera_init -> body`.

## Changes from upstream

See `git log ROS2..g1`. Known issues and the cleanup plan are in `FAST_LIO_Code_Review.md` in the project root.

## License

GPL-2.0, see `LICENSE`.
