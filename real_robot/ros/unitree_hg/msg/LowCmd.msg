# unitree_hg LowCmd (G1), as unitree_sdk2py idl/unitree_hg/msg/dds_/_LowCmd_.py. ROS topic /lowcmd = DDS rt/lowcmd.
# crc: g1_wholebody.lowlevel (checked against the SDK's crc32_core); a wrong crc makes the firmware ignore it.
uint8 mode_pr
uint8 mode_machine
MotorCmd[35] motor_cmd
uint32[4] reserve
uint32 crc
