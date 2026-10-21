#pragma once
// The voxel map (map_type: voxel), the alternative to the ikd-Tree.
//
// The world is cut into cubic voxels of `size`. Every registered scan point is added to the weighted
// statistics of its voxel (count, weight, weighted sum and sum of outer products), so no observation is
// discarded and no point is chosen by its position in the grid. The weight is 1/r^2 (range r from the
// sensor, at least 1 m): far points, seen at grazing angles, carry range-dependent offsets of centimetres
// to decimetres (results/mac/voxel_map_study), so near observations must dominate once they exist. A voxel
// also keeps its first stored_points points; from that many on it has a distribution (the weighted mean and
// covariance), and at max_points it freezes: a map that keeps adapting to the current pose drifts with it.
//
// Matching (laserMapping.cpp, h_share_model): a measurement takes the distribution among its voxel and the 6
// face neighbours that it fits best (distribution_match); without one it gets the 5 nearest stored points
// (nearest) for an upstream plane fit. Results: results/mac/voxel2_study.
//
// The voxels a scan touched also refit a plane by PCA (valid if the voxel has min_points, is thin enough and
// is a surface rather than a line). It is used by coplanar merging (merge_angle_deg > 0, off by default): a
// voxel that freezes joins the plane of its frozen planar neighbours, and matching then uses the group's
// distribution.
// TODO (map polish): the map never deletes voxels; drop those beyond det_range from the robot.
#include <algorithm>
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
    double size = 0.35;          // voxel edge [m]
    int min_points = 20;         // points needed for a plane (merging)
    double max_thickness = 0.03; // largest std. deviation across the plane [m]
    double planarity = 0.05;     // largest ratio of the smallest to the middle eigenvalue (rejects lines)
    int max_points = 30;         // a voxel stops taking points (it freezes) at this count; 0: never
    int stored_points = 10;      // points kept per voxel (at most MAX_STORED); from this many on the voxel has
                                 // a distribution
    double merge_angle_deg = 0;  // a voxel that freezes joins the plane of frozen planar neighbours within
    double merge_distance = 0.02;//   this angle and distance [m] (0: no merging)
  };
  static const int MAX_STORED = 16;

  void set_params(const Params &params) { p_ = params; }
  const Params &params() const { return p_; }
  bool empty() const { return voxels_.empty(); }
  size_t size() const { return voxels_.size(); }

  // Adds points (world frame), measured at the given ranges from the sensor, and refits the planes of the
  // voxels they fall in. Returns the number of new voxels.
  int insert(const PointVector &points, const std::vector<float> &ranges)
  {
    int created = 0;
    touched_.clear();
    frozen_now_.clear();
    for (size_t i = 0; i < points.size(); i++)
    {
      const PointType &pt = points[i];
      const V3D p(pt.x, pt.y, pt.z);
      const double r = std::max(1.0, double(ranges[i]));
      const double w = 1.0 / (r * r);
      const Eigen::Vector3i idx = index_of(p);
      const uint64_t key = pack(idx);
      Voxel &v = voxels_[key];
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
      if (v.n_pts < std::min(p_.stored_points, MAX_STORED))
        v.pts[v.n_pts++] = p.cast<float>();
      if (p_.max_points > 0 && v.n == p_.max_points)
        frozen_now_.push_back(key);
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
    if (p_.merge_angle_deg > 0)
      for (uint64_t key : frozen_now_)
        merge(key);
    return created;
  }

  // Coplanar merging: groups, and voxels in them.
  int groups() const { return group_roots_; }
  int merged_voxels() const { return merged_voxels_; }

  // The distribution (weighted mean and covariance) among the voxel of world point p and its
  // 6 face neighbours that a measurement at p with covariance meas_cov fits best, by the squared Mahalanobis
  // distance d2 = r' (cov + meas_cov)^-1 r, r = p - mean. A merged voxel offers its group's distribution
  // (grouped: the best one did). False if none of the 7 voxels has a distribution.
  bool distribution_match(const V3D &p, const M3D &meas_cov, V3D &mean, M3D &cov, double &d2,
                          bool *grouped = nullptr) const
  {
    static const int offsets[7][3] = {{0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    if (p_.stored_points <= 0)
      return false;
    const Eigen::Vector3i idx = index_of(p);
    bool found = false;
    d2 = std::numeric_limits<double>::max();
    for (const auto &o : offsets)
    {
      const auto it = voxels_.find(pack(idx + Eigen::Vector3i(o[0], o[1], o[2])));
      if (it == voxels_.end() || it->second.n < p_.stored_points)
        continue;
      const Voxel &v = it->second;
      const Group *g = v.group >= 0 ? &groups_[root(v.group)] : nullptr;
      const V3D origin = g ? g->origin : v.origin;
      const double w = g ? g->w : v.w;
      const V3D m = (g ? g->sum : v.sum) / w;
      const M3D c = (g ? g->sum_sq : v.sum_sq) / w - m * m.transpose();
      const V3D r = p - origin - m;
      const double e = r.dot((c + meas_cov).llt().solve(r));
      if (e < d2)
      {
        d2 = e;
        mean = origin + m;
        cov = c;
        found = true;
        if (grouped != nullptr)
          *grouped = g != nullptr;
      }
    }
    return found;
  }

  // The k stored points nearest to world point p, by ascending squared distance, among the
  // voxels one ring around p's voxel (27), or two rings (125) if one ring holds fewer than k points.
  void nearest(const V3D &p, int k, PointVector &out, std::vector<float> &sq_dist) const
  {
    const Eigen::Vector3i idx = index_of(p);
    const Eigen::Vector3f q = p.cast<float>();
    for (int ring = 1; ring <= 2; ring++)
    {
      out.clear();
      sq_dist.clear();
      for (int x = -ring; x <= ring; x++)
        for (int y = -ring; y <= ring; y++)
          for (int z = -ring; z <= ring; z++)
          {
            const auto it = voxels_.find(pack(idx + Eigen::Vector3i(x, y, z)));
            if (it == voxels_.end())
              continue;
            const Voxel &v = it->second;
            for (int j = 0; j < v.n_pts; j++)
            {
              const float d = (v.pts[j] - q).squaredNorm();
              if (int(sq_dist.size()) == k && d >= sq_dist.back())
                continue;
              const size_t at = std::upper_bound(sq_dist.begin(), sq_dist.end(), d) - sq_dist.begin();
              PointType pt;
              pt.x = v.pts[j](0);
              pt.y = v.pts[j](1);
              pt.z = v.pts[j](2);
              sq_dist.insert(sq_dist.begin() + at, d);
              out.insert(out.begin() + at, pt);
              if (int(sq_dist.size()) > k)
              {
                sq_dist.pop_back();
                out.pop_back();
              }
            }
          }
      if (int(out.size()) >= k)
        return;
    }
  }

  // Refits every voxel's plane, after set_params with new thresholds (debug session).
  void refit_all()
  {
    for (auto &kv : voxels_)
      fit(kv.second);
  }

  // Inspection (debug session): one voxel's statistics, in the world frame.
  struct Info
  {
    Eigen::Vector3i index;
    int n;
    double w;
    V3D mean;
    M3D cov;
    V3D normal;
    double d;
    bool valid, frozen;
  };
  template <typename F>
  void for_each(F f) const
  {
    for (const auto &kv : voxels_)
    {
      const Voxel &v = kv.second;
      Info info;
      info.index = unpack(kv.first);
      info.n = v.n;
      info.w = v.w;
      const V3D m = v.sum / v.w;
      info.mean = v.origin + m;
      info.cov = v.sum_sq / v.w - m * m.transpose();
      info.normal = v.normal;
      info.d = v.d;
      info.valid = v.valid;
      info.frozen = p_.max_points > 0 && v.n >= p_.max_points;
      f(info);
    }
  }

  // The voxel means as a cloud (for map_save): intensity = number of points, normal = plane normal
  // (zero if the voxel has no valid plane), curvature = merged group + 1 (0: none).
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
      pt.curvature = v.group >= 0 ? root(v.group) + 1 : 0;
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
    int n_pts = 0;                       // stored points
    Eigen::Vector3f pts[MAX_STORED];     // world frame
    int group = -1;                      // coplanar group (merge_angle_deg > 0), -1: none
  };

  // Coplanar frozen voxels (merge_angle_deg > 0): their summed statistics, relative to origin, and the plane
  // fitted to them. Union-find over groups: parent is the group's own index at a root.
  struct Group
  {
    int parent = 0, voxels = 0;
    V3D origin = V3D::Zero();
    double w = 0;
    V3D sum = V3D::Zero();
    M3D sum_sq = M3D::Zero();
    V3D normal = V3D::Zero();
    double d = 0;
  };

  int root(int g) const
  {
    while (groups_[g].parent != g)
      g = groups_[g].parent;
    return g;
  }

  // Adds statistics (total weight w, sums relative to origin o) to group g, moving them to g's origin.
  static void add_stats(Group &g, const V3D &o, double w, const V3D &sum, const M3D &sum_sq)
  {
    const V3D s = o - g.origin;
    g.w += w;
    g.sum += sum + w * s;
    g.sum_sq += sum_sq + sum * s.transpose() + s * sum.transpose() + w * s * s.transpose();
  }

  static void refit(Group &g)
  {
    const V3D m = g.sum / g.w;
    Eigen::SelfAdjointEigenSolver<M3D> es;
    es.computeDirect(g.sum_sq / g.w - m * m.transpose());
    g.normal = es.eigenvectors().col(0);
    g.d = -g.normal.dot(g.origin + m);
  }

  // A voxel that just froze joins the planes of its frozen planar neighbours (26) that it lies on: the
  // neighbour's group plane (its own plane without a group) is within merge_angle_deg of the voxel's normal
  // and within merge_distance of its mean. Testing against the group's plane, not pairwise, keeps a gently
  // curved surface from chaining into one plane.
  void merge(uint64_t key)
  {
    Voxel &v = voxels_.at(key);
    if (!v.valid)
      return;
    const V3D mean = v.origin + v.sum / v.w;
    const double cos_max = std::cos(p_.merge_angle_deg * M_PI / 180);
    const Eigen::Vector3i idx = unpack(key);
    for (int x = -1; x <= 1; x++)
      for (int y = -1; y <= 1; y++)
        for (int z = -1; z <= 1; z++)
        {
          if (x == 0 && y == 0 && z == 0)
            continue;
          const auto it = voxels_.find(pack(idx + Eigen::Vector3i(x, y, z)));
          if (it == voxels_.end() || it->second.n < p_.max_points || !it->second.valid)
            continue;
          Voxel &u = it->second;
          const int gu = u.group >= 0 ? root(u.group) : -1;
          const V3D n = gu >= 0 ? groups_[gu].normal : u.normal;
          const double d = gu >= 0 ? groups_[gu].d : u.d;
          if (std::fabs(n.dot(v.normal)) < cos_max || std::fabs(n.dot(mean) + d) > p_.merge_distance)
            continue;
          if (gu < 0 && std::fabs(v.normal.dot(u.origin + u.sum / u.w) + v.d) > p_.merge_distance)
            continue;  // two single voxels: each on the other's plane
          join(v, u);
        }
  }

  // Puts voxels a and b into one group: a joins b's group (a new one if b has none); two groups merge,
  // the smaller into the larger.
  void join(Voxel &a, Voxel &b)
  {
    if (b.group < 0)
    {
      Group g;
      g.parent = groups_.size();
      g.voxels = 1;
      g.origin = b.origin;
      add_stats(g, b.origin, b.w, b.sum, b.sum_sq);
      refit(g);
      b.group = g.parent;
      groups_.push_back(g);
      group_roots_++;
      merged_voxels_++;
    }
    const int gb = root(b.group), ga = a.group >= 0 ? root(a.group) : -1;
    if (ga == gb)
      return;
    if (ga < 0)
    {
      add_stats(groups_[gb], a.origin, a.w, a.sum, a.sum_sq);
      groups_[gb].voxels++;
      refit(groups_[gb]);
      a.group = gb;
      merged_voxels_++;
      return;
    }
    const int big = groups_[ga].voxels >= groups_[gb].voxels ? ga : gb, small = big == ga ? gb : ga;
    Group &G = groups_[big];
    const Group &S = groups_[small];
    add_stats(G, S.origin, S.w, S.sum, S.sum_sq);
    G.voxels += S.voxels;
    refit(G);
    groups_[small].parent = big;
    group_roots_--;
  }

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
  static Eigen::Vector3i unpack(uint64_t key)
  {
    const auto u = [](uint64_t v) { return int(v & 0x1FFFFF) - (1 << 20); };
    return Eigen::Vector3i(u(key >> 42), u(key >> 21), u(key));
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
  std::vector<uint64_t> frozen_now_;   // voxels that froze in the current insert
  std::vector<Group> groups_;
  int group_roots_ = 0, merged_voxels_ = 0;
};
