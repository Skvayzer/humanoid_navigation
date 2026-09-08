# Third-party source and licensing

This is a mixed-source deployment repository. Original copyright notices and
component licenses are retained; there is no blanket relicensing of third-party
code. Do not assume that a public repository makes every component MIT-licensed.

| Component | Included location | License / provenance |
| --- | --- | --- |
| Fast-LIO and bundled ikd-tree/IKFoM code | `src/fast_lio` | Retained GPL-2.0 and per-file notices; inspect `LICENSE` |
| NDT-OMP | `src/ndt_omp_ros2` | BSD-2-Clause; retained license |
| LiDAR localization | `src/lidar_localization_ros2` | BSD-2-Clause; retained license |
| Livox message interface | `src/livox_ros_driver2` | BSD-3-Clause package declaration; interface-only |
| Livox visualization adapter | `visualization/livox` | BSD-3-Clause package declaration |
| Unitree SDK 2 Python | `third_party/unitree_sdk2_python` | BSD-3-Clause; exact source commit in PROVENANCE |
| rosbridge overlay | `visualization/rosbridge` | BSD-3-Clause; original license and per-file notices |
| Open3D native CPU extension | Release runtime archive | MIT; `vendor/licenses/Open3D-MIT.txt`; upstream source linked in PROVENANCE |
| CycloneDDS | Release runtime + source archives | EPL-2.0 / EDL-1.0; license and notices in `vendor/licenses` |
| rmw_cyclonedds | Release runtime + source archives | Apache-2.0; license in `vendor/licenses` |
| ROS 2 / Nav2 / PCL and OS libraries | Installed inside Docker by apt | Their upstream licenses; not copied as source into this repository |

The SDK includes small upstream CRC native libraries for ARM64/x86; these remain
with its source as shipped upstream. Do not execute its unrelated robot examples.

For repository-original files without an existing license notice, no additional
open-source license is selected by this publication. Contact the repository owner
for licensing terms. This does not restrict rights already granted by the
third-party licenses above.
