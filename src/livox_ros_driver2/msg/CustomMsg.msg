# Exact wire-compatible definition used by the G1 host livox_ros_driver2.
std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
CustomPoint[] points
