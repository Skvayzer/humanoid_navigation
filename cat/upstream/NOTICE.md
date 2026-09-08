# CAT perception provenance

Upstream: https://github.com/GalaxyGeneralRobotics/Click-and-Traverse

Pinned commit: `866ba392f1c1e84b92ad75fa66550f26e8af8e48`

`cat_preprocess` in `cat/core.py` adapts the morphology and vertical fill from
`deploy/scripts/exp_dis_pf/octomap_bridge.py`, licensed Apache-2.0 (see LICENSE).
It preserves the 5x5x5 closing, two iterations, 3x3x3 neighbor test, upward
fill from index 20 and downward fill below index 5. It removes import-time
shared memory attachment, policy/SDK dependencies, ROS side effects, and fake
corner obstacles. Shape validation and a pure-function interface were added.

Grid shape/resolution and the OctoMap sensor-model values are taken from that
script and `deploy/Click-and-Traverse-SLAM/lidar_ws/src/octomap_mapping/`
`octomap_server/launch/octomap_mapping.launch.xml` at the same commit.

The native OctoMap wrapper, ROS adapter, tests and launchers are new integration
code, not the upstream deployment controller. This is not a complete CAT port
or a CAT policy observation producer: no SDF, boundary/guidance fields,
proprioception, ONNX inference, policy shared memory or joint commands yet.
