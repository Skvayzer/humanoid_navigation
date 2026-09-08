#ifndef LIDAR_LOCALIZATION_G2_NDT_CANDIDATE_SCORE_HPP_
#define LIDAR_LOCALIZATION_G2_NDT_CANDIDATE_SCORE_HPP_

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pclomp/ndt_omp.h>
#include <pclomp/ndt_omp_impl.hpp>
#include <pclomp/voxel_grid_covariance_omp_impl.hpp>

namespace lidar_localization
{
namespace g2_ndt
{

using Cloud = pcl::PointCloud<pcl::PointXYZI>;
using CloudPtr = Cloud::Ptr;

struct G2NdtScoreParams
{
  double ndt_resolution{1.0};
  double ndt_step_size{0.1};
  double transform_epsilon{0.01};
  int max_iterations{30};
  int num_threads{1};
  double scan_voxel_leaf_size{1.0};
  double target_voxel_leaf_size{1.0};
  double local_map_radius{150.0};
  std::size_t min_target_points{100};
};

struct G2NdtScoreResult
{
  double fitness{std::numeric_limits<double>::infinity()};
  bool converged{false};
  std::size_t target_point_count{0};
  std::size_t source_point_count{0};
  // Pose from ndt.getFinalTransformation() when converged; otherwise the seed
  // pose passed to score() (x, y, z, roll, pitch, yaw).
  double refined_x{std::numeric_limits<double>::quiet_NaN()};
  double refined_y{std::numeric_limits<double>::quiet_NaN()};
  double refined_z{std::numeric_limits<double>::quiet_NaN()};
  double refined_roll{std::numeric_limits<double>::quiet_NaN()};
  double refined_pitch{std::numeric_limits<double>::quiet_NaN()};
  double refined_yaw{std::numeric_limits<double>::quiet_NaN()};
};

inline bool has_point_field(
  const std::vector<pcl::PCLPointField> & fields,
  const std::string & name)
{
  return std::any_of(fields.begin(), fields.end(), [&](const auto & field) {
    return field.name == name;
  });
}

inline CloudPtr load_cloud_xyzi(const std::string & path)
{
  pcl::PCLPointCloud2 raw;
  const std::string lower = [&]() {
    std::string copy = path;
    std::transform(copy.begin(), copy.end(), copy.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return copy;
  }();
  int result = -1;
  if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".pcd") {
    result = pcl::io::loadPCDFile(path, raw);
  } else if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".ply") {
    result = pcl::io::loadPLYFile(path, raw);
  } else {
    throw std::runtime_error("unsupported point cloud suffix: " + path);
  }
  if (result != 0) {
    throw std::runtime_error("failed to load point cloud: " + path);
  }

  CloudPtr cloud(new Cloud());
  if (has_point_field(raw.fields, "intensity")) {
    pcl::fromPCLPointCloud2(raw, *cloud);
  } else {
    pcl::PointCloud<pcl::PointXYZ> xyz;
    pcl::fromPCLPointCloud2(raw, xyz);
    cloud->reserve(xyz.size());
    for (const auto & point : xyz.points) {
      pcl::PointXYZI xyzi;
      xyzi.x = point.x;
      xyzi.y = point.y;
      xyzi.z = point.z;
      xyzi.intensity = 0.0F;
      cloud->push_back(xyzi);
    }
  }
  return cloud;
}

inline CloudPtr voxel_downsample(const CloudPtr & input, double leaf_size)
{
  if (leaf_size <= 0.0 || input->empty()) {
    return input;
  }
  CloudPtr filtered(new Cloud());
  pcl::VoxelGrid<pcl::PointXYZI> voxel;
  voxel.setLeafSize(
    static_cast<float>(leaf_size),
    static_cast<float>(leaf_size),
    static_cast<float>(leaf_size));
  voxel.setInputCloud(input);
  voxel.filter(*filtered);
  return filtered;
}

inline CloudPtr crop_map_xy(const CloudPtr & map, double x, double y, double radius)
{
  if (radius <= 0.0) {
    return map;
  }
  const double r2 = radius * radius;
  CloudPtr cropped(new Cloud());
  cropped->reserve(map->size() / 10);
  for (const auto & point : map->points) {
    const double dx = static_cast<double>(point.x) - x;
    const double dy = static_cast<double>(point.y) - y;
    if (dx * dx + dy * dy <= r2) {
      cropped->push_back(point);
    }
  }
  return cropped;
}

inline Eigen::Matrix4f pose_matrix(
  double x, double y, double z, double roll, double pitch, double yaw)
{
  Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
  matrix.block<3, 3>(0, 0) =
    (Eigen::AngleAxisf(static_cast<float>(yaw), Eigen::Vector3f::UnitZ()) *
    Eigen::AngleAxisf(static_cast<float>(pitch), Eigen::Vector3f::UnitY()) *
    Eigen::AngleAxisf(static_cast<float>(roll), Eigen::Vector3f::UnitX()))
    .toRotationMatrix();
  matrix(0, 3) = static_cast<float>(x);
  matrix(1, 3) = static_cast<float>(y);
  matrix(2, 3) = static_cast<float>(z);
  return matrix;
}

inline Eigen::Matrix4f pose_matrix(double x, double y, double z, double yaw)
{
  return pose_matrix(x, y, z, 0.0, 0.0, yaw);
}

inline CloudPtr cloud_from_xyz(
  const double * xyz,
  std::size_t point_count,
  double scan_voxel_leaf_size)
{
  CloudPtr cloud(new Cloud());
  cloud->reserve(point_count);
  for (std::size_t i = 0; i < point_count; ++i) {
    pcl::PointXYZI point;
    point.x = static_cast<float>(xyz[3 * i]);
    point.y = static_cast<float>(xyz[3 * i + 1]);
    point.z = static_cast<float>(xyz[3 * i + 2]);
    point.intensity = 0.0F;
    cloud->push_back(point);
  }
  return voxel_downsample(cloud, scan_voxel_leaf_size);
}

class G2NdtCandidateScorer
{
public:
  G2NdtCandidateScorer(const std::string & map_path, G2NdtScoreParams params)
  : params_(params), full_map_(load_cloud_xyzi(map_path))
  {
  }

  G2NdtScoreResult score(
    const CloudPtr & source,
    double x,
    double y,
    double z,
    double yaw) const
  {
    return score(source, x, y, z, 0.0, 0.0, yaw);
  }

  G2NdtScoreResult score(
    const CloudPtr & source,
    double x,
    double y,
    double z,
    double roll,
    double pitch,
    double yaw) const
  {
    G2NdtScoreResult result;
    result.refined_x = x;
    result.refined_y = y;
    result.refined_z = z;
    result.refined_roll = roll;
    result.refined_pitch = pitch;
    result.refined_yaw = yaw;
    result.source_point_count = source ? source->size() : 0;
    if (!source || source->empty()) {
      return result;
    }

    CloudPtr target = crop_map_xy(full_map_, x, y, params_.local_map_radius);
    result.target_point_count = target->size();
    if (target->size() < params_.min_target_points) {
      return result;
    }
    target = voxel_downsample(target, params_.target_voxel_leaf_size);

    pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    ndt.setResolution(params_.ndt_resolution);
    ndt.setStepSize(params_.ndt_step_size);
    ndt.setTransformationEpsilon(params_.transform_epsilon);
    ndt.setMaximumIterations(params_.max_iterations);
    ndt.setNumThreads(std::max(1, params_.num_threads));
    ndt.setInputTarget(target);
    ndt.setInputSource(source);

    const Eigen::Matrix4f init = pose_matrix(x, y, z, roll, pitch, yaw);
    Cloud output;
    ndt.align(output, init);
    result.converged = ndt.hasConverged();
    const double fitness = ndt.getFitnessScore();
    result.fitness = std::isfinite(fitness) ? fitness : std::numeric_limits<double>::infinity();
    if (result.converged) {
      const Eigen::Matrix4f final = ndt.getFinalTransformation();
      result.refined_x = static_cast<double>(final(0, 3));
      result.refined_y = static_cast<double>(final(1, 3));
      result.refined_z = static_cast<double>(final(2, 3));
      result.refined_roll = std::atan2(
        static_cast<double>(final(2, 1)),
        static_cast<double>(final(2, 2)));
      result.refined_pitch = std::asin(std::clamp(
          -static_cast<double>(final(2, 0)), -1.0, 1.0));
      result.refined_yaw = std::atan2(
        static_cast<double>(final(1, 0)),
        static_cast<double>(final(0, 0)));
    }
    return result;
  }

  G2NdtScoreResult score_xyz(
    const double * xyz,
    std::size_t point_count,
    double x,
    double y,
    double z,
    double yaw) const
  {
    return score(cloud_from_xyz(xyz, point_count, params_.scan_voxel_leaf_size), x, y, z, yaw);
  }

  G2NdtScoreResult score_xyz(
    const double * xyz,
    std::size_t point_count,
    double x,
    double y,
    double z,
    double roll,
    double pitch,
    double yaw) const
  {
    return score(
      cloud_from_xyz(xyz, point_count, params_.scan_voxel_leaf_size),
      x, y, z, roll, pitch, yaw);
  }

private:
  G2NdtScoreParams params_;
  CloudPtr full_map_;
};

}  // namespace g2_ndt
}  // namespace lidar_localization

#endif  // LIDAR_LOCALIZATION_G2_NDT_CANDIDATE_SCORE_HPP_
