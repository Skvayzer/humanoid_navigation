# Publication validation — 2026-09-08

## Passed

- Fresh anonymous GitHub clone and public dependency download; archive and native
  library SHA-256 checks matched.
- Seven local/CI test groups: Python syntax, Bash syntax, XML, captured robot
  algorithm/config hashes, gateway limits/nonfinite values, mandatory operator
  arming guard, and non-overwriting/idempotent shortcut installation.
- Direct read-only hash comparison with the G1 for Fast-LIO laserMapping.cpp,
  mid360_g1.yaml and g1_nav2_planning.yaml: identical.
- Full native ARM64 Docker build on a clean GitHub-hosted runner: four source
  packages compiled (livox_ros_driver2, fast_lio, ndt_omp_ros2,
  lidar_localization_ros2).
- Resulting container ran with **network disabled**, read-only root and dropped
  capabilities; validated NumPy 1.17.4, SciPy 1.3.3, Open3D 0.16.0, rclpy, and
  Fast-LIO/NDT/Nav2 package discovery plus the Livox message interface.

[ARM64 build and offline validation log](https://github.com/Skvayzer/humanoid_navigation/actions/runs/34241117936)

The build tested commit `2a5ca50a7132fe66aa33634e18b8bacc89fa7768`. Subsequent
publication commits added the deployed diagnostic script, documentation,
line-ending preservation and quieter bootstrap output; they did not change the
Dockerfile, estimator code, Nav2 configuration or container entrypoint.

The CI-local image ID was
`sha256:11992ee0062aab57dca65b14740afdbd64400907c271f62ab67afeba2328f491`.
This is an audit identifier, **not a pullable registry reference**; the image was
not uploaded to a container registry.

## Not performed

No software was installed/changed on the existing robot. No stack was started,
no locomotion mode was changed, and no motor command was sent for publication.
The portable packaging was not physically retested on a second robot. Host venv
creation, sensor compatibility, DDS discovery, map registration and actual motion
must be verified on the target before arming. This validation is not a safety
certification or a guarantee of navigation performance.
