#pragma once
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

using namespace std;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// preprocess.lidar_type: which message the lidar topic carries.
enum LID_TYPE
{
  LIVOX_CUSTOM = 1,  // livox_ros_driver2/CustomMsg
  POINTCLOUD2 = 2    // sensor_msgs/PointCloud2 with a per-point time field (any other value is read as 2)
};
// preprocess.timestamp_unit: unit of the PointCloud2 time field.
enum TIME_UNIT
{
  SEC = 0,
  MS = 1,
  US = 2,
  NS = 3
};

// Turns one lidar message (one full scan) into the cloud FAST_LIO works on: a random share
// (point_keep_ratio) of the points farther than `blind`, with its time after the header stamp in
// `curvature` [ms].
class Preprocess
{
  public:
  void process(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void process(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloudXYZI::Ptr &pcl_out);

  int lidar_type = LIVOX_CUSTOM;
  double point_keep_ratio = 1.0;  // share of the valid points kept; random, but reproducible per scan
  uint64_t random_seed = 0;       // varies the random choices (0: seeded by the scan stamp only)
  double blind = 0.01;
  int N_SCANS = 6;               // CustomMsg: points on a line >= N_SCANS are dropped
  string time_field = "time";    // PointCloud2: name of the per-point time field
  int time_unit = NS;            // PointCloud2: its unit (TIME_UNIT)
  bool time_absolute = false;    // PointCloud2: absolute time (e.g. livox_ros_driver2 `timestamp`),
                                 // otherwise time after the header stamp (e.g. G1 vendor `time`)

  private:
  bool keep(uint64_t seed, size_t i) const;
  void update_keep_threshold();
  uint64_t keep_threshold_ = 0;   // point_keep_ratio as a 53-bit integer threshold
  void livox_handler(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg, PointCloudXYZI &out);
  void pointcloud2_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloudXYZI &out);
};
