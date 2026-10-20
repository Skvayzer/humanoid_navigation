cmake_minimum_required(VERSION 3.10)
project(fast_lio)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# C++17: required by rclcpp from Humble on and by PCL >= 1.14; fine on Foxy (gcc 9).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O3 -pthread -fexceptions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# FAST_LIO writes its timing log (runtime_pos_log_enable) to <source>/Log/.
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# OpenMP threads for the per-point nearest-search/residual loop: the num_threads parameter, whose
# default is set here by core count. (Upstream enabled OpenMP on x86 only, which left ARM computers such
# as the G1's Jetson single-threaded.)
include(ProcessorCount)
ProcessorCount(N)
if(N GREATER 4)
  set(MP_PROC_NUM 3)
elseif(N GREATER 3)
  set(MP_PROC_NUM 2)
else()
  set(MP_PROC_NUM 1)
endif()
add_definitions(-DMP_PROC_NUM=${MP_PROC_NUM})

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
if(OpenMP_CXX_FOUND)
  add_definitions(-DMP_EN)
endif()
message(STATUS "${N} cores: OpenMP ${OpenMP_CXX_FOUND}, ${MP_PROC_NUM} threads by default (num_threads)")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(Eigen3 REQUIRED)
# filters: pcl::VoxelGrid is precompiled in libpcl_filters
find_package(PCL REQUIRED COMPONENTS common io filters)

add_executable(fastlio_mapping src/laserMapping.cpp src/preprocess.cpp include/ikd-Tree/ikd_Tree.cpp)
target_include_directories(fastlio_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${PCL_INCLUDE_DIRS}
)
# ikd-Tree rebuilds large subtrees in a background thread, which makes the odometry depend on thread
# timing in rare cases. Regression checks build with -DIKD_SYNC_REBUILD=ON to get deterministic output.
option(IKD_SYNC_REBUILD "Rebuild ikd-Tree subtrees synchronously (deterministic, for regression checks)" OFF)
if(IKD_SYNC_REBUILD)
  target_compile_definitions(fastlio_mapping PRIVATE Multi_Thread_Rebuild_Point_Num=1000000000)
endif()
target_link_libraries(fastlio_mapping ${PCL_LIBRARIES} Eigen3::Eigen)
# Offline replay (--replay) reads zlib-compressed .g1replay datasets.
find_package(ZLIB)
if(ZLIB_FOUND)
  target_compile_definitions(fastlio_mapping PRIVATE FASTLIO_REPLAY)
  target_link_libraries(fastlio_mapping ZLIB::ZLIB)
else()
  message(WARNING "zlib not found: fastlio_mapping is built without --replay")
endif()
ament_target_dependencies(fastlio_mapping
  rclcpp geometry_msgs nav_msgs sensor_msgs std_msgs std_srvs tf2_ros livox_ros_driver2)

install(TARGETS fastlio_mapping DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
