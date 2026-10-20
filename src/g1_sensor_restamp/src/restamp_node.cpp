// Re-stamps the G1 vendor Mid-360 IMU stream onto the sensor's own regular sample grid.
//   in:  /utlidar/imu_livox_mid360  host stamps; the vendor process stalls up to ~14 ms at every
//                                   LiDAR scan and then releases the queued samples in a burst
//   out: /g1/imu                    same messages, header.stamp = sample time, repeats dropped
// If the input goes quiet for max_gap the node warns once. When the data returns with a stamp jump
// the grid restarts at once; output continues while it settles.
// The algorithm is in restamper.hpp; the measurements are in G1_Robot_Environment.md, section 5.3.1.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "g1_sensor_restamp/restamper.hpp"

namespace g1_sensor_restamp
{

using Clock = std::chrono::steady_clock;

class RestampNode : public rclcpp::Node
{
public:
  explicit RestampNode(const rclcpp::NodeOptions & options)
  : Node("g1_sensor_restamp", options)
  {
    in_topic_ = declare_parameter<std::string>("imu_in", "/utlidar/imu_livox_mid360");
    out_topic_ = declare_parameter<std::string>("imu_out", "/g1/imu");
    const bool reliable = declare_parameter<bool>("reliable_input", true);
    acc_scale_ = declare_parameter<double>("acc_scale", 1.0);
    ImuParams p;
    p.period = declare_parameter<double>("period", p.period);
    p.creep = declare_parameter<double>("creep", p.creep);
    p.max_gap = declare_parameter<double>("max_gap", p.max_gap);
    p.drop_duplicates = declare_parameter<bool>("drop_duplicates", p.drop_duplicates);
    max_gap_ = p.max_gap;

    restamper_ = std::make_unique<ImuRestamper<sensor_msgs::msg::Imu>>(
      p, [this](double t, const sensor_msgs::msg::Imu & m) {publish(t, m);});

    // RELIABLE out, so that both reliable and best-effort subscribers can connect.
    pub_ = create_publisher<sensor_msgs::msg::Imu>(
      out_topic_, rclcpp::QoS(rclcpp::KeepLast(200)).reliable());
    rclcpp::QoS qin(rclcpp::KeepLast(200));
    if (reliable) {qin.reliable();} else {qin.best_effort();}
    sub_ = create_subscription<sensor_msgs::msg::Imu>(
      in_topic_, qin, [this](sensor_msgs::msg::Imu::ConstSharedPtr m) {on_imu(*m);});
    watchdog_ = create_wall_timer(std::chrono::milliseconds(50), [this]() {check_input();});
    stats_timer_ = create_wall_timer(std::chrono::seconds(10), [this]() {log_stats();});

    RCLCPP_INFO(
      get_logger(), "%s -> %s (%s input)", in_topic_.c_str(), out_topic_.c_str(),
      reliable ? "reliable" : "best-effort");
  }

private:
  void on_imu(const sensor_msgs::msg::Imu & m)
  {
    const auto now = Clock::now();
    if (quiet_) {
      RCLCPP_INFO(
        get_logger(), "IMU data back after %.2f s",
        std::chrono::duration<double>(now - last_rx_).count());
      quiet_ = false;
    }
    last_rx_ = now;
    have_rx_ = true;

    const int64_t ns =
      static_cast<int64_t>(m.header.stamp.sec) * 1000000000LL + m.header.stamp.nanosec;
    if (!have_epoch_) {
      epoch_ns_ = ns;
      last_ns_ = ns;
      have_epoch_ = true;
    }
    const uint64_t restarts = restamper_->stats().restarts;
    const bool was_settling = restamper_->settling();
    const auto & a = m.linear_acceleration;
    const auto & w = m.angular_velocity;
    restamper_->push(1e-9 * static_cast<double>(ns - epoch_ns_), {a.x, a.y, a.z, w.x, w.y, w.z}, m);
    if (restamper_->stats().restarts != restarts) {
      RCLCPP_WARN(
        get_logger(), "stamp jump of %+.3f s: grid restarted, settling",
        1e-9 * static_cast<double>(ns - last_ns_));
    }
    if (was_settling && !restamper_->settling()) {
      RCLCPP_INFO(get_logger(), "grid settled");
    }
    last_ns_ = ns;
  }

  void check_input()
  {
    if (!have_rx_ || quiet_) {return;}
    const double silent = std::chrono::duration<double>(Clock::now() - last_rx_).count();
    if (silent > max_gap_) {
      RCLCPP_WARN(get_logger(), "no IMU data on %s for %.2f s", in_topic_.c_str(), silent);
      quiet_ = true;
    }
  }

  void publish(double t, const sensor_msgs::msg::Imu & m)
  {
    auto out = std::make_unique<sensor_msgs::msg::Imu>(m);
    const int64_t ns = epoch_ns_ + std::llround(t * 1e9);
    out->header.stamp.sec = static_cast<int32_t>(ns / 1000000000LL);
    out->header.stamp.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
    if (acc_scale_ != 1.0) {
      out->linear_acceleration.x *= acc_scale_;
      out->linear_acceleration.y *= acc_scale_;
      out->linear_acceleration.z *= acc_scale_;
      if (out->linear_acceleration_covariance[0] != -1.0) {   // -1 marks "unknown"
        for (double & c : out->linear_acceleration_covariance) {c *= acc_scale_ * acc_scale_;}
      }
    }
    pub_->publish(std::move(out));
  }

  void log_stats()
  {
    const Stats & s = restamper_->stats();
    if (s.received == 0) {
      RCLCPP_WARN(get_logger(), "no messages on %s yet", in_topic_.c_str());
      return;
    }
    RCLCPP_INFO(
      get_logger(),
      "in %llu out %llu | held %llu, max delay %.1f ms | lost %llu (%llu events) | repeats %llu | "
      "restarts %llu",
      (unsigned long long)s.received, (unsigned long long)s.emitted, (unsigned long long)s.held,
      1e3 * s.max_delay, (unsigned long long)s.lost, (unsigned long long)s.loss_events,
      (unsigned long long)s.duplicates, (unsigned long long)s.restarts);
  }

  std::string in_topic_, out_topic_;
  double acc_scale_ = 1.0;
  double max_gap_ = 0.1;
  bool have_epoch_ = false;
  int64_t epoch_ns_ = 0;
  int64_t last_ns_ = 0;
  bool have_rx_ = false;
  bool quiet_ = false;
  Clock::time_point last_rx_;
  std::unique_ptr<ImuRestamper<sensor_msgs::msg::Imu>> restamper_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr watchdog_, stats_timer_;
};

}  // namespace g1_sensor_restamp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<g1_sensor_restamp::RestampNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
