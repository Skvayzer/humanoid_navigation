# Operator commands

Use the commands installed from this checkout. They are not system services and
do not automatically resume after reboot. Only the human operator may change the
G1 locomotion/FSM mode. Keep physical emergency control ready.

## Localization and visualization

```bash
slam-start                         # default: teaching_lab_aligned.pcd
# or: slam-start-localize my_map.pcd
foxglove-start                     # Rosbridge at ws://<robot-ip>:9090
```

Keep still during initialization. In Foxglove, select fixed frame `map` and check
`/g1_localization/initial_map`, `/g1_slam/cloud_registered` and TF. Wait for NDT to
be active and verify the pose is actually correct; "running" is not validation.

## Navigation preview — no arming

```bash
nav-start-plan
```

This disarms the gateway, then starts the same Nav2 controller used for motion.
Publish `geometry_msgs/msg/PoseStamped` to `/g1_nav/goal_pose`, frame `map`, with a
valid quaternion. Observe `/g1_nav/plan`, `/g1_nav/cmd_vel_preview`, and
`/g1_nav/local_costmap/costmap` plus its update topic. The global map is
`/g1_nav/global_costmap/costmap`. No gateway is required just for preview.

## Supervised real motion — separate manual decision

```bash
killnav
nav-start-safe                    # starts the gateway DISARMED
g1-motion-status
```

Inspect live obstacles, map alignment, forward-axis convention, command limits
and emergency controls. Only after the operator has selected the correct robot
mode and is ready to supervise:

```bash
I_AM_SUPERVISING=1 g1-motion-arm
```

Arming has a one-second zero-command settling period and requires a fresh command.
The gateway may issue zero-velocity requests even while disarmed; "disarmed"
means it must not pass nonzero velocity, not that it is disconnected from DDS.
Send a nearby goal only when ready. Limits: forward 0.20 m/s, no reverse/lateral,
yaw ±0.20 rad/s, linear ramp 0.10 m/s², yaw ramp 0.20 rad/s², 0.30 s watchdog.
Current Nav2 rotates toward the path first; final goal-yaw spin is disabled.

## Stop

```bash
killnav                           # disarm + stop Nav2; SLAM stays running
g1-motion-stop                    # disarm only; gateway continues sending zero
slam-stop                         # mode-aware SLAM stop; mapping saves first
```

`killnav` is a software stop, **not a certified emergency stop**. It depends on
the computer/network/SDK being responsive. A crane is not a substitute for
validated robot support and physical emergency control.

## Collect a map

```bash
killnav
slam-stop
slam-start-map
# Move/scan only under the operator's control.
slam-stop-map                     # save must succeed before removal
```

Maps use timestamped filenames in `$HOME/g1_slam_runtime/maps` by default. Do not
use `docker rm -f` or `docker kill` during collection: they can lose unsaved data.
There is no arbitrary shutdown delay, but writing the map can take time.

## Logs

```bash
sudo docker logs g1-slam-backup-20260713
tail -n 80 "$HOME/g1_slam_runtime/logs/fast_lio.log"
tail -n 80 "$HOME/g1_slam_runtime/logs/ndt_localizer.log"
tail -n 80 "$HOME/g1_slam_runtime/logs/nav/planning.log"
tail -n 80 "$HOME/g1_slam_runtime/logs/nav/g1_motion_bridge.log"
tail -n 80 "$HOME/g1_slam_runtime/logs/rosbridge/bridge.log"
```

Adjust paths/container name if site.env overrides the defaults.
