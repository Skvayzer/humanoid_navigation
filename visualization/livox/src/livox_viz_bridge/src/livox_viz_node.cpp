#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/header.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace
{
struct VizPoint
{
  float x;
  float y;
  float z;
  float intensity;
  float time;
  std::uint16_t ring;
  std::uint8_t tag;
};

std::int64_t stamp_to_ns(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}
}  // namespace

class LivoxVizBridge final : public rclcpp::Node
{
public:
  LivoxVizBridge()
  : Node("livox_viz_bridge")
  {
    const auto lidar_input = declare_parameter<std::string>("lidar_input", "/livox/lidar");
    const auto imu_input = declare_parameter<std::string>("imu_input", "/livox/imu");
    const auto cloud_output =
      declare_parameter<std::string>("cloud_output", "/g1_viz/livox_points");
    const auto marker_output =
      declare_parameter<std::string>("marker_output", "/g1_viz/livox_imu_markers");
    scan_period_ms_ = static_cast<int>(std::clamp<std::int64_t>(
      declare_parameter<std::int64_t>("scan_period_ms", 100), 20, 1000));
    max_points_ = static_cast<std::size_t>(std::max<std::int64_t>(
      declare_parameter<std::int64_t>("max_points_per_scan", 200000), 1000));
    gyro_scale_ = declare_parameter<double>("gyro_arrow_scale", 5.0);

    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      cloud_output, rclcpp::SensorDataQoS().keep_last(2));
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      marker_output, rclcpp::QoS(2).best_effort());

    static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    geometry_msgs::msg::TransformStamped lidar_transform;
    lidar_transform.header.stamp = now();
    lidar_transform.header.frame_id = "body";
    lidar_transform.child_frame_id = "livox_frame";
    lidar_transform.transform.translation.x = -0.011;
    lidar_transform.transform.translation.y = -0.02329;
    lidar_transform.transform.translation.z = 0.04412;
    lidar_transform.transform.rotation.w = 1.0;
    static_tf_broadcaster_->sendTransform(lidar_transform);

    lidar_sub_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
      lidar_input, rclcpp::SensorDataQoS().keep_last(64),
      std::bind(&LivoxVizBridge::lidar_callback, this, std::placeholders::_1));
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_input, rclcpp::SensorDataQoS().keep_last(8),
      std::bind(&LivoxVizBridge::imu_callback, this, std::placeholders::_1));

    cloud_timer_ = create_wall_timer(
      std::chrono::milliseconds(scan_period_ms_),
      std::bind(&LivoxVizBridge::publish_cloud, this));
    marker_timer_ = create_wall_timer(
      std::chrono::milliseconds(50), std::bind(&LivoxVizBridge::publish_imu_markers, this));

    points_.reserve(std::min<std::size_t>(max_points_, 20000));
    RCLCPP_INFO(
      get_logger(),
      "Read-only visualization bridge: %s -> %s (%d ms), %s -> %s",
      lidar_input.c_str(), cloud_output.c_str(), scan_period_ms_, imu_input.c_str(),
      marker_output.c_str());
  }

private:
  void lidar_callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (batch_start_ns_ == 0) {
      batch_start_ns_ = stamp_to_ns(msg->header.stamp);
      cloud_header_ = msg->header;
    }

    const auto packet_ns = stamp_to_ns(msg->header.stamp);
    if (packet_ns < batch_start_ns_ || packet_ns - batch_start_ns_ > 2000000000LL) {
      points_.clear();
      batch_start_ns_ = packet_ns;
      cloud_header_ = msg->header;
    }

    const auto room = max_points_ > points_.size() ? max_points_ - points_.size() : 0;
    const auto count = std::min<std::size_t>(room, msg->points.size());
    const double packet_offset = static_cast<double>(packet_ns - batch_start_ns_) * 1e-9;
    for (std::size_t index = 0; index < count; ++index) {
      const auto & point = msg->points[index];
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
        continue;
      }
      points_.push_back(VizPoint{
        point.x,
        point.y,
        point.z,
        static_cast<float>(point.reflectivity),
        static_cast<float>(packet_offset + static_cast<double>(point.offset_time) * 1e-9),
        static_cast<std::uint16_t>(point.line),
        point.tag});
    }
    if (count < msg->points.size()) {
      ++truncated_packets_;
    }
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_imu_ = *msg;
  }

  void publish_cloud()
  {
    std::vector<VizPoint> points;
    std_msgs::msg::Header header;
    std::uint64_t truncated = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (points_.empty()) {
        return;
      }
      points.swap(points_);
      header = cloud_header_;
      batch_start_ns_ = 0;
      truncated = std::exchange(truncated_packets_, 0);
    }

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = header;
    cloud.height = 1;
    cloud.width = static_cast<std::uint32_t>(points.size());
    cloud.is_bigendian = false;
    cloud.is_dense = true;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(
      7,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "intensity", 1, sensor_msgs::msg::PointField::FLOAT32,
      "time", 1, sensor_msgs::msg::PointField::FLOAT32,
      "ring", 1, sensor_msgs::msg::PointField::UINT16,
      "tag", 1, sensor_msgs::msg::PointField::UINT8);
    modifier.resize(points.size());

    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> intensity(cloud, "intensity");
    sensor_msgs::PointCloud2Iterator<float> time(cloud, "time");
    sensor_msgs::PointCloud2Iterator<std::uint16_t> ring(cloud, "ring");
    sensor_msgs::PointCloud2Iterator<std::uint8_t> tag(cloud, "tag");
    for (const auto & point : points) {
      *x = point.x;
      *y = point.y;
      *z = point.z;
      *intensity = point.intensity;
      *time = point.time;
      *ring = point.ring;
      *tag = point.tag;
      ++x;
      ++y;
      ++z;
      ++intensity;
      ++time;
      ++ring;
      ++tag;
    }

    cloud_pub_->publish(cloud);
    if (truncated != 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Visualization buffer reached its point cap; truncated packets=%lu",
        static_cast<unsigned long>(truncated));
    }
  }

  visualization_msgs::msg::Marker make_arrow(
    const std_msgs::msg::Header & header, int id, const std::string & name,
    double x, double y, double z, double red, double green, double blue) const
  {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "livox_imu";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.025;
    marker.scale.y = 0.055;
    marker.scale.z = 0.08;
    marker.color.r = static_cast<float>(red);
    marker.color.g = static_cast<float>(green);
    marker.color.b = static_cast<float>(blue);
    marker.color.a = 1.0F;
    marker.lifetime.sec = 0;
    marker.lifetime.nanosec = 200000000U;
    geometry_msgs::msg::Point origin;
    geometry_msgs::msg::Point endpoint;
    endpoint.x = x;
    endpoint.y = y;
    endpoint.z = z;
    marker.points = {origin, endpoint};
    marker.text = name;
    return marker;
  }

  void publish_imu_markers()
  {
    std::optional<sensor_msgs::msg::Imu> imu;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      imu = latest_imu_;
    }
    if (!imu) {
      return;
    }

    visualization_msgs::msg::MarkerArray array;
    array.markers.push_back(make_arrow(
      imu->header, 0, "gravity_direction (-linear_acceleration)",
      -imu->linear_acceleration.x, -imu->linear_acceleration.y, -imu->linear_acceleration.z,
      0.95, 0.1, 0.1));
    array.markers.push_back(make_arrow(
      imu->header, 1, "angular_velocity",
      imu->angular_velocity.x * gyro_scale_, imu->angular_velocity.y * gyro_scale_,
      imu->angular_velocity.z * gyro_scale_, 1.0, 0.45, 0.05));
    marker_pub_->publish(array);
  }

  std::mutex mutex_;
  std::vector<VizPoint> points_;
  std::optional<sensor_msgs::msg::Imu> latest_imu_;
  std_msgs::msg::Header cloud_header_;
  std::int64_t batch_start_ns_{0};
  std::size_t max_points_{200000};
  std::uint64_t truncated_packets_{0};
  int scan_period_ms_{100};
  double gyro_scale_{5.0};

  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr cloud_timer_;
  rclcpp::TimerBase::SharedPtr marker_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LivoxVizBridge>());
  rclcpp::shutdown();
  return 0;
}
