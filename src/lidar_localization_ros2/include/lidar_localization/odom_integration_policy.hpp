#ifndef LIDAR_LOCALIZATION_ODOM_INTEGRATION_POLICY_HPP_
#define LIDAR_LOCALIZATION_ODOM_INTEGRATION_POLICY_HPP_

#include <Eigen/Geometry>

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "lidar_localization/pose_finite.hpp"

namespace lidar_localization
{

enum class OdomAdmissionStatus
{
  kAccepted = 0,
  kDisabled,
  kWaitingForInitialPose,
  kMissingCurrentPose,
  kIntervalTooLarge,
  kIntervalNegative,
  kCurrentPoseNonFinite,
};

struct OdomAdmissionInput
{
  bool use_odom{false};
  bool initial_pose_received{false};
  bool has_current_pose{false};
  double dt_odom_sec{0.0};
  bool current_pose_finite{true};
  double max_odom_interval_sec{1.0};
};

struct OdomAdmissionDecision
{
  OdomAdmissionStatus status{OdomAdmissionStatus::kAccepted};
  bool accepted{true};
};

inline OdomAdmissionDecision decideOdomAdmission(const OdomAdmissionInput & input)
{
  OdomAdmissionDecision decision;
  if (!input.use_odom) {
    decision.status = OdomAdmissionStatus::kDisabled;
    decision.accepted = false;
    return decision;
  }
  if (!input.initial_pose_received) {
    decision.status = OdomAdmissionStatus::kWaitingForInitialPose;
    decision.accepted = false;
    return decision;
  }
  if (!input.has_current_pose) {
    decision.status = OdomAdmissionStatus::kMissingCurrentPose;
    decision.accepted = false;
    return decision;
  }
  if (!input.current_pose_finite) {
    decision.status = OdomAdmissionStatus::kCurrentPoseNonFinite;
    decision.accepted = false;
    return decision;
  }
  if (input.dt_odom_sec > input.max_odom_interval_sec) {
    decision.status = OdomAdmissionStatus::kIntervalTooLarge;
    decision.accepted = false;
    return decision;
  }
  if (input.dt_odom_sec < 0.0) {
    decision.status = OdomAdmissionStatus::kIntervalNegative;
    decision.accepted = false;
    return decision;
  }
  return decision;
}

inline geometry_msgs::msg::Pose integrateBodyFrameTwist(
  const geometry_msgs::msg::Pose & pose,
  const geometry_msgs::msg::Twist & twist,
  double dt_sec)
{
  Eigen::Quaterniond world_q_body{
    pose.orientation.w,
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z};
  world_q_body.normalize();

  const Eigen::Vector3d angular_velocity_body{
    twist.angular.x, twist.angular.y, twist.angular.z};
  const double rotation_angle = angular_velocity_body.norm() * dt_sec;
  Eigen::Quaterniond body_q_delta = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond body_q_half_delta = Eigen::Quaterniond::Identity();
  if (rotation_angle > 1.0e-12) {
    const Eigen::Vector3d rotation_axis = angular_velocity_body.normalized();
    body_q_delta = Eigen::AngleAxisd(rotation_angle, rotation_axis);
    body_q_half_delta = Eigen::AngleAxisd(0.5 * rotation_angle, rotation_axis);
  }

  const Eigen::Vector3d linear_velocity_body{
    twist.linear.x, twist.linear.y, twist.linear.z};
  const Eigen::Vector3d delta_position_world =
    (world_q_body * body_q_half_delta) * (linear_velocity_body * dt_sec);
  Eigen::Quaterniond next_world_q_body = world_q_body * body_q_delta;
  next_world_q_body.normalize();

  geometry_msgs::msg::Pose result = pose;
  result.position.x += delta_position_world.x();
  result.position.y += delta_position_world.y();
  result.position.z += delta_position_world.z();
  result.orientation.x = next_world_q_body.x();
  result.orientation.y = next_world_q_body.y();
  result.orientation.z = next_world_q_body.z();
  result.orientation.w = next_world_q_body.w();
  return result;
}

}  // namespace lidar_localization

#endif  // LIDAR_LOCALIZATION_ODOM_INTEGRATION_POLICY_HPP_
