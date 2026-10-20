cmake_minimum_required(VERSION 3.5)
project(g1_sensor_restamp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(include)

add_executable(restamp_node src/restamp_node.cpp)
ament_target_dependencies(restamp_node rclcpp sensor_msgs)

# ROS-free offline evaluation on recorded stamp traces and synthetic streams
add_executable(restamp_eval src/restamp_eval.cpp)

install(TARGETS restamp_node restamp_eval DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/record_stamp_trace.py scripts/record_scan_info.py
  scripts/record_restamp_check.py scripts/plot_restamp_check.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
