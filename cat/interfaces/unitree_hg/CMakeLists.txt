cmake_minimum_required(VERSION 3.5)
project(unitree_hg)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/IMUState.msg" "msg/MotorState.msg" "msg/LowState.msg")
ament_export_dependencies(rosidl_default_runtime)
ament_package()
