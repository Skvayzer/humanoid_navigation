cmake_minimum_required(VERSION 3.5)
project(livox_viz_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(livox_viz_node src/livox_viz_node.cpp)
ament_target_dependencies(livox_viz_node
  geometry_msgs
  livox_ros_driver2
  rclcpp
  sensor_msgs
  std_msgs
  tf2_ros
  visualization_msgs
)

install(TARGETS livox_viz_node DESTINATION lib/${PROJECT_NAME})

ament_package()
