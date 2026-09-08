# humanoid_navigation

Containerized SLAM, map localization and Nav2 navigation for a Unitree G1.
Captured **directly from the deployed robot on 2026-09-08**, with the working G1
fixes—not reconstructed from an older desktop checkout.

The original stack came from an Agibot X2 implementation. This repository captures
its deployed **Foxy G1 port**. It does not install Click-and-Traverse, MPPI, ROS
Humble, or a new low-level locomotion policy.

## What is included

- Fast-LIO mapping/odometry and the G1 LiDAR–IMU extrinsic correction.
- NDT-OMP map localization, gravity/floor diagnostics, map conversion tools.
- Nav2 NavigateToPose behavior tree, Navfn planner, Regulated Pure Pursuit,
  planar pose/odometry bridge, live-LiDAR local costmap and Foxglove path relay.
- Multi-stage Docker build and runtime scripts, preserving the vendor ROS graph.
- Host-side Unitree motion gateway: manual arming, speed/ramp limits, watchdog,
  forward-only motion, and `killnav`.
- Host rosbridge compatibility overlay, Livox visualization adapter and shortcuts.
- Exact deployed SDK source; checksum-verified ARM64 native dependencies in the
  [snapshot release](https://github.com/Skvayzer/humanoid_navigation/releases/tag/robot-snapshot-20260908).

## Architecture

```text
Existing robot Livox driver
  └─ /livox/lidar + /livox/imu
       └─ one Foxy Docker container
            Fast-LIO → NDT localization → Nav2 → /g1_nav/cmd_vel_preview
                          live cloud ───→ local costmap
                                                 │
                         host safety gateway (manual arm)
                                                 │
                              Unitree G1 high-level SetVelocity

Host rosbridge + Livox adapter → Foxglove visualization
```

The gateway and visualization are host processes, matching the working deployment.
Host networking allows vendor discovery. No vendor workspace is bind-mounted or
modified. This isolation protects dependencies; it does **not** isolate DDS
commands, TF names, CPU/memory use, or guarantee physical safety.

## Start here

1. [Install on another robot](docs/INSTALL.md).
2. [Operator startup, preview, arm and stop commands](docs/COMMANDS.md).
3. [Maps, frames, calibration and obstacle behavior](docs/MAPS_AND_FRAMES.md).
4. [Snapshot provenance and packaging changes](PROVENANCE.md).
5. [Third-party licenses](THIRD_PARTY.md).
6. [Fresh ARM64 build and validation results](docs/VALIDATION.md).

```bash
git clone https://github.com/Skvayzer/humanoid_navigation.git
cd humanoid_navigation
./scripts/verify_source.sh
./scripts/fetch_runtime.sh
```

Read the installation guide before building or starting anything. Clone into a new
directory; the shortcut installer refuses to overwrite existing project commands.

## Safety and portability

Target baseline: Ubuntu 20.04, ARM64, ROS 2 Foxy, G1/MID360 with the tested vendor
DDS/driver conventions. Other robots need calibration and a compatible command
adapter. The gateway never changes locomotion/FSM mode. Only the operator does
that. Default limits are 0.20 m/s forward and ±0.20 rad/s yaw, with no lateral or
reverse motion. Software stopping is not a replacement for physical emergency
control.

Preview is disarmed. Live obstacles are enabled in the captured local costmap.
The Foxy controller is **not MPPI**: it does not reproduce every behavior of the
original Humble/X2 navigation stack.

Public Git/releases contain no teaching-lab maps, recordings, logs, credentials or
runtime arm tokens. Supply a private PCD and matching occupancy map, or collect
your own. The retained lab initial pose is an example, not a universal default.

Foxy is end-of-life. The repository preserves application source and exact native
runtime hashes; upstream apt availability is still needed for rebuilding. Source
checks are automatic; the ARM64 container build workflow validates the build and
offline imports without contacting a robot. Neither is a physical-motion test.
