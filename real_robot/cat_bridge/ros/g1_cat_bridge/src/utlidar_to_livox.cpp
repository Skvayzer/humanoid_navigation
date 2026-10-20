// Read-only adapter: G1 vendor MID360 stream -> livox_ros_driver2 topics used by
// humanoid_navigation (Fast-LIO and CAT).
//
// Input  /utlidar/cloud_livox_mid360  PointCloud2 {x,y,z,intensity:f32, ring:u16,
//        time:f32 = ns since header.stamp}, native (upside-down) sensor axes.
// Output /livox/lidar  CustomMsg with points rolled 180 deg about X, i.e. the
//        convention of the driver the repo was tuned on (its extrinsic_R and
//        CAT's diag(1,-1,-1) undo it exactly once).
// IMU    /utlidar/imu_livox_mid360 -> /livox/imu unchanged (native axes, g units).
//
// Subscriptions are BEST_EFFORT so this node can never back-pressure the vendor
// publisher. Nothing is sent to the LiDAR, the robot or its services.
#include <cmath>
#include <cstring>
#include <string>

#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

using livox_ros_driver2::msg::CustomMsg;
using livox_ros_driver2::msg::CustomPoint;
using sensor_msgs::msg::Imu;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

namespace {
int field_offset(const PointCloud2 &m, const std::string &name, uint8_t type) {
  for (const auto &f : m.fields)
    if (f.name == name) return (f.datatype == type && f.count == 1) ? static_cast<int>(f.offset) : -2;
  return -1;
}
template <typename T> T load(const uint8_t *p) { T v; std::memcpy(&v, p, sizeof(T)); return v; }
}  // namespace

class UtlidarToLivox : public rclcpp::Node {
 public:
  UtlidarToLivox() : Node("utlidar_to_livox", "g1_cat_bridge") {
    const auto in_cloud = declare_parameter<std::string>("input_cloud", "/utlidar/cloud_livox_mid360");
    const auto in_imu = declare_parameter<std::string>("input_imu", "/utlidar/imu_livox_mid360");
    const auto out_cloud = declare_parameter<std::string>("output_cloud", "/livox/lidar");
    const auto out_imu = declare_parameter<std::string>("output_imu", "/livox/imu");
    frame_ = declare_parameter<std::string>("frame_id", "livox_frame");
    roll_ = declare_parameter<bool>("roll_points_180", true);
    cloud_pub_ = create_publisher<CustomMsg>(out_cloud, rclcpp::QoS(10).reliable());
    imu_pub_ = create_publisher<Imu>(out_imu, rclcpp::QoS(200).reliable());
    cloud_sub_ = create_subscription<PointCloud2>(in_cloud, rclcpp::SensorDataQoS().keep_last(5),
        [this](PointCloud2::UniquePtr m) { on_cloud(*m); });
    imu_sub_ = create_subscription<Imu>(in_imu, rclcpp::SensorDataQoS().keep_last(200),
        [this](Imu::UniquePtr m) { ++imus_; imu_pub_->publish(std::move(m)); });
    timer_ = create_wall_timer(std::chrono::seconds(10), [this] { report(); });
    RCLCPP_INFO(get_logger(), "READ-ONLY adapter %s -> %s, %s -> %s, roll_points_180=%s",
                in_cloud.c_str(), out_cloud.c_str(), in_imu.c_str(), out_imu.c_str(), roll_ ? "true" : "false");
  }

 private:
  void on_cloud(const PointCloud2 &m) {
    const int ox = field_offset(m, "x", PointField::FLOAT32), oy = field_offset(m, "y", PointField::FLOAT32),
              oz = field_offset(m, "z", PointField::FLOAT32), oi = field_offset(m, "intensity", PointField::FLOAT32),
              orr = field_offset(m, "ring", PointField::UINT16), ot = field_offset(m, "time", PointField::FLOAT32);
    if (ox < 0 || oy < 0 || oz < 0 || oi < 0 || orr < 0 || ot < 0 || m.is_bigendian || m.header.frame_id != frame_ ||
        m.point_step < 22 || m.data.size() < static_cast<size_t>(m.point_step) * m.width * m.height) {
      ++rejected_;
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
          "unexpected cloud layout/frame '%s' (step %u); expected x,y,z,intensity,ring,time in %s",
          m.header.frame_id.c_str(), m.point_step, frame_.c_str());
      return;
    }
    const size_t n = static_cast<size_t>(m.width) * m.height;
    auto out = std::make_unique<CustomMsg>();
    out->header = m.header;
    out->timebase = static_cast<uint64_t>(m.header.stamp.sec) * 1000000000ULL + m.header.stamp.nanosec;
    out->lidar_id = 0;
    out->points.reserve(n);
    const float s = roll_ ? -1.0f : 1.0f;
    size_t skipped = 0;
    for (size_t i = 0; i < n; ++i) {
      const uint8_t *p = m.data.data() + i * m.point_step;
      const float x = load<float>(p + ox), y = load<float>(p + oy), z = load<float>(p + oz);
      const float t = load<float>(p + ot), in = load<float>(p + oi);
      const uint16_t ring = load<uint16_t>(p + orr);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(t) ||
          t < 0.0f || t > 2.0e8f || ring > 255) { ++skipped; continue; }
      CustomPoint q;
      q.offset_time = static_cast<uint32_t>(std::lround(t));
      q.x = x; q.y = s * y; q.z = s * z;
      q.reflectivity = static_cast<uint8_t>(std::lround(std::min(255.0f, std::max(0.0f, in))));
      q.tag = 0;  // normal return (CAT and Fast-LIO accept tag&0x30 in {0x00,0x10})
      q.line = static_cast<uint8_t>(ring);
      out->points.push_back(q);
    }
    out->point_num = static_cast<uint32_t>(out->points.size());
    skipped_ += skipped;
    if (out->point_num == 0) { ++rejected_; return; }
    ++clouds_; points_ += out->point_num;
    cloud_pub_->publish(std::move(out));
  }

  void report() {
    RCLCPP_INFO(get_logger(), "last 10 s: clouds=%zu (%.1f Hz, %.0f pts avg) imu=%zu (%.0f Hz) rejected=%zu skipped_pts=%zu",
                clouds_, clouds_ / 10.0, clouds_ ? double(points_) / clouds_ : 0.0, imus_, imus_ / 10.0, rejected_, skipped_);
    clouds_ = points_ = imus_ = rejected_ = skipped_ = 0;
  }

  std::string frame_;
  bool roll_{true};
  size_t clouds_{0}, points_{0}, imus_{0}, rejected_{0}, skipped_{0};
  rclcpp::Publisher<CustomMsg>::SharedPtr cloud_pub_;
  rclcpp::Publisher<Imu>::SharedPtr imu_pub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<Imu>::SharedPtr imu_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<UtlidarToLivox>());
  rclcpp::shutdown();
  return 0;
}
