#pragma once
// Scan downsampling into patches (scan_patch.min_points > 0), the alternative to one centroid per voxel.
//
// The deskewed scan is cut into voxels of filter_size_surf (lidar frame). A voxel with fewer than min_points
// points keeps all of them: in sparse regions every point is a measurement. A fuller voxel becomes one
// measurement, its centroid, which also carries the covariance of its points for the distribution residual
// in laserMapping.cpp (h_share_model). Results: results/mac/patch_study, results/mac/voxel2_study.
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <common_lib.h>

struct ScanPatch
{
  int n = 1;                 // points: 1 for a single point
  M3D cov = M3D::Zero();     // covariance of the points (lidar frame)
};

// Returns the number of patches (centroids) in out.
inline int voxelize_scan(const PointCloudXYZI &in, double leaf, int min_points, PointCloudXYZI &out,
                         std::vector<ScanPatch> &patches)
{
  const size_t N = in.points.size();
  std::vector<std::pair<uint64_t, uint32_t>> keys(N);  // voxel key (21 bits per axis), point index
  const auto u = [leaf](float v) { return uint64_t(int64_t(std::floor(v / leaf)) + (1 << 20)) & 0x1FFFFF; };
  for (size_t i = 0; i < N; i++)
  {
    const PointType &p = in.points[i];
    keys[i] = {(u(p.x) << 42) | (u(p.y) << 21) | u(p.z), uint32_t(i)};
  }
  std::sort(keys.begin(), keys.end());
  out.clear();
  patches.clear();
  int count = 0;
  for (size_t b = 0, e; b < N; b = e)
  {
    for (e = b + 1; e < N && keys[e].first == keys[b].first; e++)
      ;
    const int n = e - b;
    if (n < min_points)
    {
      for (size_t k = b; k < e; k++)
      {
        out.push_back(in.points[keys[k].second]);
        patches.push_back(ScanPatch());
      }
      continue;
    }
    ScanPatch sp;
    V3D mean = V3D::Zero();
    double intensity = 0;
    for (size_t k = b; k < e; k++)
    {
      const PointType &p = in.points[keys[k].second];
      mean += V3D(p.x, p.y, p.z);
      intensity += p.intensity;
    }
    mean /= n;
    for (size_t k = b; k < e; k++)
    {
      const PointType &p = in.points[keys[k].second];
      const V3D q = V3D(p.x, p.y, p.z) - mean;
      sp.cov += q * q.transpose();
    }
    sp.cov /= n;
    sp.n = n;
    PointType c;
    c.x = mean(0);
    c.y = mean(1);
    c.z = mean(2);
    c.intensity = intensity / n;
    out.push_back(c);
    patches.push_back(sp);
    count++;
  }
  return count;
}
