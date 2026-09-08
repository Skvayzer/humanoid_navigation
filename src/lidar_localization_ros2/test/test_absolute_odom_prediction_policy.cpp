#include "lidar_localization/absolute_odom_prediction_policy.hpp"

#include <cassert>
#include <cmath>
#include <deque>

namespace ll = lidar_localization;

namespace
{

Eigen::Matrix4f transform(double x, double y, double z, double roll, double pitch, double yaw)
{
  Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
  result.block<3, 3>(0, 0) =
    (Eigen::AngleAxisf(static_cast<float>(yaw), Eigen::Vector3f::UnitZ()) *
    Eigen::AngleAxisf(static_cast<float>(pitch), Eigen::Vector3f::UnitY()) *
    Eigen::AngleAxisf(static_cast<float>(roll), Eigen::Vector3f::UnitX())).toRotationMatrix();
  result.block<3, 1>(0, 3) = Eigen::Vector3f{
    static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
  return result;
}

template<typename DerivedA, typename DerivedB>
void assert_matrix_near(
  const Eigen::MatrixBase<DerivedA> & actual,
  const Eigen::MatrixBase<DerivedB> & expected,
  float tolerance = 1.0e-5f)
{
  assert((actual - expected).cwiseAbs().maxCoeff() < tolerance);
}

void test_map_to_odom_round_trip_preserves_full_pose()
{
  const auto map_to_base = transform(4.0, -2.0, 0.7, 0.2, -0.3, 1.1);
  const auto odom_to_base = transform(1.0, 3.0, -0.2, -0.1, 0.4, -0.6);
  const auto map_to_odom = ll::mapToOdomFromPoses(map_to_base, odom_to_base);
  assert_matrix_near(
    ll::predictMapPoseFromAbsoluteOdom(map_to_odom, odom_to_base),
    map_to_base);
}

void test_rigid_blend_interpolates_translation_and_rotation()
{
  const auto current = transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  const auto measured = transform(2.0, 4.0, 6.0, 0.0, 0.0, M_PI / 2.0);
  const auto blended = ll::blendRigidTransforms(current, measured, 0.25);

  assert_matrix_near(blended.block<3, 1>(0, 3), Eigen::Vector3f(0.5f, 1.0f, 1.5f));
  const Eigen::Quaternionf rotation(blended.block<3, 3>(0, 0));
  const Eigen::Quaternionf expected(
    Eigen::AngleAxisf(static_cast<float>(M_PI / 8.0), Eigen::Vector3f::UnitZ()));
  assert(std::abs(rotation.normalized().dot(expected.normalized())) > 1.0f - 1.0e-5f);
}

void test_nearest_pose_is_bounded_by_time_delta()
{
  const std::deque<ll::TimedPoseMatrix> history{
    {10.0, transform(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
    {10.1, transform(2.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
    {10.2, transform(3.0, 0.0, 0.0, 0.0, 0.0, 0.0)}};
  Eigen::Matrix4f selected = Eigen::Matrix4f::Identity();
  assert(ll::nearestTimedPose(history, 10.11, 0.02, &selected));
  assert(std::abs(selected(0, 3) - 2.0f) < 1.0e-6f);
  assert(!ll::nearestTimedPose(history, 10.15, 0.02, &selected));
}

void test_pose_conversion_rejects_invalid_quaternion()
{
  geometry_msgs::msg::Pose pose;
  pose.orientation.x = 0.0;
  pose.orientation.y = 0.0;
  pose.orientation.z = 0.0;
  pose.orientation.w = 0.0;
  Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
  assert(!ll::poseMatrixFromPose(pose, &matrix));

  pose.orientation.w = 2.0;
  pose.position.x = 1.0;
  assert(ll::poseMatrixFromPose(pose, &matrix));
  assert(std::abs(matrix(0, 3) - 1.0f) < 1.0e-6f);
  assert_matrix_near(matrix.block<3, 3>(0, 0), Eigen::Matrix3f::Identity());
}

}  // namespace

int main()
{
  test_map_to_odom_round_trip_preserves_full_pose();
  test_rigid_blend_interpolates_translation_and_rotation();
  test_nearest_pose_is_bounded_by_time_delta();
  test_pose_conversion_rejects_invalid_quaternion();
  return 0;
}
