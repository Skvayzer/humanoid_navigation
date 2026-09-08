#ifndef LIDAR_LOCALIZATION_ABSOLUTE_ODOM_PREDICTION_POLICY_HPP_
#define LIDAR_LOCALIZATION_ABSOLUTE_ODOM_PREDICTION_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>

namespace lidar_localization
{

struct TimedPoseMatrix
{
  double stamp_sec{0.0};
  Eigen::Matrix4f pose_matrix{Eigen::Matrix4f::Identity()};
};

inline bool poseMatrixFromPose(
  const geometry_msgs::msg::Pose & pose,
  Eigen::Matrix4f * pose_matrix)
{
  if (pose_matrix == nullptr) {
    return false;
  }

  const Eigen::Vector3f translation{
    static_cast<float>(pose.position.x),
    static_cast<float>(pose.position.y),
    static_cast<float>(pose.position.z)};
  Eigen::Quaternionf rotation{
    static_cast<float>(pose.orientation.w),
    static_cast<float>(pose.orientation.x),
    static_cast<float>(pose.orientation.y),
    static_cast<float>(pose.orientation.z)};
  if (!translation.allFinite() || !rotation.coeffs().allFinite() || rotation.norm() < 1.0e-6f) {
    return false;
  }

  rotation.normalize();
  *pose_matrix = Eigen::Matrix4f::Identity();
  pose_matrix->block<3, 3>(0, 0) = rotation.toRotationMatrix();
  pose_matrix->block<3, 1>(0, 3) = translation;
  return pose_matrix->allFinite();
}

inline Eigen::Matrix4f mapToOdomFromPoses(
  const Eigen::Matrix4f & map_to_base,
  const Eigen::Matrix4f & odom_to_base)
{
  return map_to_base * odom_to_base.inverse();
}

inline Eigen::Matrix4f predictMapPoseFromAbsoluteOdom(
  const Eigen::Matrix4f & map_to_odom,
  const Eigen::Matrix4f & odom_to_base)
{
  return map_to_odom * odom_to_base;
}

inline Eigen::Matrix4f blendRigidTransforms(
  const Eigen::Matrix4f & current,
  const Eigen::Matrix4f & measured,
  double gain)
{
  const float clamped_gain = static_cast<float>(std::clamp(gain, 0.0, 1.0));
  if (clamped_gain <= 0.0f || !measured.allFinite()) {
    return current;
  }
  if (clamped_gain >= 1.0f || !current.allFinite()) {
    return measured;
  }

  Eigen::Quaternionf current_rotation(current.block<3, 3>(0, 0));
  Eigen::Quaternionf measured_rotation(measured.block<3, 3>(0, 0));
  if (
    !current_rotation.coeffs().allFinite() || !measured_rotation.coeffs().allFinite() ||
    current_rotation.norm() < 1.0e-6f || measured_rotation.norm() < 1.0e-6f)
  {
    return current;
  }
  current_rotation.normalize();
  measured_rotation.normalize();

  Eigen::Matrix4f blended = Eigen::Matrix4f::Identity();
  blended.block<3, 3>(0, 0) =
    current_rotation.slerp(clamped_gain, measured_rotation).normalized().toRotationMatrix();
  blended.block<3, 1>(0, 3) =
    (1.0f - clamped_gain) * current.block<3, 1>(0, 3) +
    clamped_gain * measured.block<3, 1>(0, 3);
  return blended;
}

inline bool nearestTimedPose(
  const std::deque<TimedPoseMatrix> & history,
  double stamp_sec,
  double max_time_delta_sec,
  Eigen::Matrix4f * pose_matrix)
{
  if (
    history.empty() || pose_matrix == nullptr || !std::isfinite(stamp_sec) ||
    !std::isfinite(max_time_delta_sec) || max_time_delta_sec < 0.0)
  {
    return false;
  }

  const TimedPoseMatrix * nearest = nullptr;
  double nearest_delta = std::numeric_limits<double>::infinity();
  for (auto it = history.rbegin(); it != history.rend(); ++it) {
    const double delta = std::abs(it->stamp_sec - stamp_sec);
    if (delta < nearest_delta) {
      nearest = &*it;
      nearest_delta = delta;
    }
    if (it->stamp_sec < stamp_sec && delta > nearest_delta) {
      break;
    }
  }

  if (nearest == nullptr || nearest_delta > max_time_delta_sec || !nearest->pose_matrix.allFinite()) {
    return false;
  }
  *pose_matrix = nearest->pose_matrix;
  return true;
}

}  // namespace lidar_localization

#endif  // LIDAR_LOCALIZATION_ABSOLUTE_ODOM_PREDICTION_POLICY_HPP_
