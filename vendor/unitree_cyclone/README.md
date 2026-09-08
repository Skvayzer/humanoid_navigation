These ARM64 runtime libraries are copied read-only from the Unitree G1's
`cyclonedds_ws` installation. The vendor Livox driver uses this custom Cyclone
DDS build; stock ROS 2 Foxy Cyclone crashes when joining its DDS domain.

Expected files:

- `libddsc.so`
- `librmw_cyclonedds_cpp.so`
