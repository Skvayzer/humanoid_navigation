#pragma once
// Voxel map of plane statistics, the alternative to the ikd-Tree (map_type: voxel).
//
// The world is cut into cubic voxels of `size`. Every registered scan point is added to the weighted
// statistics of its voxel (count, weight, weighted sum and sum of outer products), so no observation is
// discarded and no point is chosen by its position in the grid. The weight is 1/r^2 (range r from the
// sensor, at least 1 m): far points, seen at grazing angles, carry range-dependent offsets of centimetres
// to decimetres (results/mac/voxel_map_study), so near observations must dominate a plane once they exist. The voxels a scan touched then refit their plane by PCA: the normal is the
// eigenvector of the smallest eigenvalue, and the plane is valid if the voxel has enough points, is thin
// enough, and is a surface rather than a line. A query point takes the valid plane it lies closest to
// among its own voxel and the 6 face neighbours: a few hash lookups, no kNN search, no plane fit.
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>
#include <Eigen/Eigenvalues>
#include <common_lib.h>

class VoxelMap
{
  public:
  struct Params
  {
    double size = 0.25;          // voxel edge [m]
    int min_points = 20;         // points needed for a plane
    double max_thickness = 0.03; // largest std. deviation across the plane [m]
    double planarity = 0.05;     // largest ratio of the smallest to the middle eigenvalue (rejects lines)
    int max_points = 30;         // a voxel stops taking points (its plane is frozen) at this count; 0: never
  };

  void set_params(const Params &params) { p_ = params; }
  bool empty() const { return voxels_.empty(); }
  size_t size() const { return voxels_.size(); }

  // Adds points (world frame), measured at the given ranges from the sensor, and refits the planes of the
  // voxels they fall in. Returns the number of new voxels.
  int insert(const PointVector &points, const std::vector<float> &ranges)
  {
    int created = 0;
    touched_.clear();
    for (size_t i = 0; i < points.size(); i++)
    {
      const PointType &pt = points[i];
      const V3D p(pt.x, pt.y, pt.z);
      const double r = std::max(1.0, double(ranges[i]));
      const double w = 1.0 / (r * r);
      const Eigen::Vector3i idx = index_of(p);
      Voxel &v = voxels_[pack(idx)];
      if (p_.max_points > 0 && v.n >= p_.max_points)
        continue;  // frozen: a map that keeps adapting to the current pose drifts with it
      if (v.n == 0)
      {
        v.origin = idx.cast<double>() * p_.size;  // statistics are kept relative to the voxel corner
        created++;
      }
      const V3D q = p - v.origin;
      v.n++;
      v.w += w;
      v.sum += w * q;
      v.sum_sq += w * q * q.transpose();
      if (!v.touched)
      {
        v.touched = true;
        touched_.push_back(&v);  // element pointers stay valid when an unordered_map rehashes
      }
    }
    for (Voxel *v : touched_)
    {
      fit(*v);
      v->touched = false;
    }
    return created;
  }

  // The plane (unit normal n, offset d: n.x + d = 0) for a world point: the valid plane of its voxel or
  // of a face neighbour that the point lies closest to. False if none of them has a valid plane.
  bool plane_for(const V3D &p, VF(4) &plane) const
  {
    static const int offsets[7][3] = {{0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const Eigen::Vector3i idx = index_of(p);
    const Voxel *best = nullptr;
    double best_dist = std::numeric_limits<double>::max();
    for (const auto &o : offsets)
    {
      const auto it = voxels_.find(pack(idx + Eigen::Vector3i(o[0], o[1], o[2])));
      if (it == voxels_.end() || !it->second.valid)
        continue;
      const double dist = std::fabs(it->second.normal.dot(p) + it->second.d);
      if (dist < best_dist)
      {
        best_dist = dist;
        best = &it->second;
      }
    }
    if (best == nullptr)
      return false;
    plane << float(best->normal(0)), float(best->normal(1)), float(best->normal(2)), float(best->d);
    return true;
  }

  // The voxel means as a cloud (for map_save): intensity = number of points, normal = plane normal
  // (zero if the voxel has no valid plane).
  void to_cloud(PointCloudXYZI &out) const
  {
    out.clear();
    for (const auto &kv : voxels_)
    {
      const Voxel &v = kv.second;
      const V3D mean = v.origin + v.sum / v.w;
      PointType pt;
      pt.x = mean(0);
      pt.y = mean(1);
      pt.z = mean(2);
      pt.intensity = v.n;
      pt.normal_x = v.valid ? v.normal(0) : 0;
      pt.normal_y = v.valid ? v.normal(1) : 0;
      pt.normal_z = v.valid ? v.normal(2) : 0;
      out.push_back(pt);
    }
  }

  private:
  struct Voxel
  {
    V3D origin = V3D::Zero();
    int n = 0;                 // points
    double w = 0;              // their total weight
    V3D sum = V3D::Zero();     // weighted, relative to origin
    M3D sum_sq = M3D::Zero();
    V3D normal = V3D::Zero();
    double d = 0;
    bool valid = false, touched = false;
  };

  Eigen::Vector3i index_of(const V3D &p) const
  {
    return Eigen::Vector3i(int(std::floor(p(0) / p_.size)), int(std::floor(p(1) / p_.size)), int(std::floor(p(2) / p_.size)));
  }

  // 21 bits per axis: +-2^20 voxels around the origin.
  static uint64_t pack(const Eigen::Vector3i &i)
  {
    const auto u = [](int v) { return uint64_t(v + (1 << 20)) & 0x1FFFFF; };
    return (u(i(0)) << 42) | (u(i(1)) << 21) | u(i(2));
  }

  void fit(Voxel &v) const
  {
    v.valid = false;
    if (v.n < p_.min_points)
      return;
    const V3D mean = v.sum / v.w;
    const M3D cov = v.sum_sq / v.w - mean * mean.transpose();
    Eigen::SelfAdjointEigenSolver<M3D> es;
    es.computeDirect(cov);
    const V3D l = es.eigenvalues();  // ascending
    if (l(0) > p_.max_thickness * p_.max_thickness || l(0) > p_.planarity * l(1))
      return;
    v.normal = es.eigenvectors().col(0);
    v.d = -v.normal.dot(v.origin + mean);
    v.valid = true;
  }

  Params p_;
  std::unordered_map<uint64_t, Voxel> voxels_;
  std::vector<Voxel *> touched_;
};
