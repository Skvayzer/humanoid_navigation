#pragma once
// Relocalization on a prior voxel map (localization mode): the robot's pose on the map from a submap of its last
// scans, near a guess or anywhere on the map.
//
// The prior map is gravity-aligned (z up) with a single floor, its origin the mapping run's sensor (IMU) at the
// start. The submap is in the robot's level frame L: origin at the IMU, z up, x along the IMU's heading (the
// filter's roll and pitch are removed), so only x, y, z and yaw of L on the map are unknown. z comes from the
// floor: the map's floor height plus the IMU's height above the floor seen in the submap, or the mapping sensor's
// height (z = 0, the same robot) if either is not seen. So the search is over x, y and yaw:
//   1. branch and bound over translations on the map's voxel grid for every yaw step (about one voxel at
//      max_range), within xy_window / yaw_window of a guess or over the whole map, best first. A pose's score is
//      the sum of its points' voxel values (2 occupied, 1 next to an occupied voxel: a grid pose is up to half a
//      voxel off, so a wall point often lands beside the wall), using the points band_low to band_high above the
//      floor (floor and ceiling points fit almost any x, y and yaw); a node of 2^k x 2^k translations is bounded by
//      the level-k values, the maximum over each 2^k x 2^k block (an exact upper bound), so the first
//      full-resolution pose taken is the best scoring one. The search collects the best distinct poses;
//   2. with z_window > 0, for robots whose floor estimate can be wrong (an upright sensor high on a robot sees the
//      floor only metres away, desk tops nearer), the search scores all points (the band above the floor is not
//      known) and each pose gets the z (within z_window) at which the most points, floor and ceiling included,
//      fall into occupied voxels; then Gauss-Newton on x, y, z and yaw refines them with the filter's
//      point-to-distribution residuals (VoxelMap::distribution_match, chi-square gate);
//   3. the best refined pose is accepted if enough points match it and no other distinct pose matches nearly as
//      many: an ambiguous place (e.g. a symmetric room) is reported, not guessed.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>
#include <string>
#include <vector>
#include <voxel_map.h>

class Relocalizer
{
  // The submap's floor: the densest layer of points between these depths below the IMU [m], within FLOOR_RADIUS
  static constexpr double FLOOR_DEPTH_MIN = 0.2, FLOOR_DEPTH_MAX = 3.0;
  static constexpr double FLOOR_RADIUS = 20.0;        // [m]
  static constexpr double FLOOR_BIN = 0.02;           // layer thickness [m]
  static constexpr int FLOOR_MIN_POINTS = 50;         // in the densest layer, else no floor
  static constexpr double FLOOR_REFINE = 0.04;        // points this close to the layer give its height [m]
  // The map's floor: horizontal planar voxels below the map's origin (the mapping sensor)
  static constexpr double MAP_FLOOR_MARGIN = 0.2;     // only voxels this far below the origin [m]
  static constexpr double MAP_FLOOR_TILT_DEG = 10.0;  // largest angle of a horizontal voxel's normal to z
  static constexpr int MAP_FLOOR_MIN_VOXELS = 10;
  static constexpr double MAP_FLOOR_BIN = 0.05;       // [m]
  static constexpr double MAP_FLOOR_SHARE = 0.3;      // the lowest bin with this share of the strongest one's points
  static constexpr double MAP_FLOOR_REFINE = 0.1;     // voxels this close to the bin give the height [m]
  // The search
  static constexpr size_t MIN_POINTS = 100;           // submap points needed (in range, and in the score band)
  static constexpr int OCCUPIED_POINTS = 2;           // a map voxel with this many points is occupied
  static constexpr uint8_t OCCUPIED = 2, NEIGHBOUR = 1;   // the score of a point in an occupied voxel / next to one
  static constexpr int MAX_LEVEL = 10;                // pyramid levels: blocks up to 2^10 voxels
  static constexpr double DISTINCT_XY = 1.0, DISTINCT_YAW_DEG = 10.0;   // closer search poses are one candidate
  static constexpr double OTHER_XY = 0.5, OTHER_YAW_DEG = 5.0;   // a farther refined pose is another place (ambiguity)
  // The refinement
  static constexpr int REFINE_ITERATIONS = 15;
  static constexpr int REFINE_MIN_MATCHED = 20;       // points, else the iterations stop
  static constexpr double POINT_COV = 0.001;          // as the filter's LASER_POINT_COV
  static constexpr double GATE = 11.34;               // as the filter's D2D_GATE (chi-square, 3 dof, 99 %)
  static constexpr double CONVERGED_XYZ = 1e-4, CONVERGED_YAW = 1e-5;   // [m], [rad]

  public:
  struct Params
  {
    double xy_window = 2.0;       // search half-width around a guess [m]
    double yaw_window = 45.0;     // [deg]
    double z_window = 0.0;        // z checked around the floor-based start [m] (0: none)
    double z_step = 0.1;
    double max_range = 25.0;      // submap points used, from the IMU [m]; also sets the yaw step
    double band_low = 0.2;        // heights above the floor of the points scored in the search [m] (floor and
    double band_high = 2.5;       //   ceiling points fit almost anywhere)
    int search_points = 300;      // submap points scored in the search (evenly subsampled)
    int refine_points = 2000;     // and in the refinement
    int candidates = 5;           // distinct search poses refined
    double min_share = 0.5;       // the search stops below this share of the best pose's score
    double accept_matched = 0.8;  // share of the refinement points the pose must match
    double max_ambiguity = 0.9;   // a distinct pose matching this share of the best's matches rejects the result
  };
  struct Pose
  {
    double x = 0, y = 0, z = 0, yaw = 0;   // L in the map frame (yaw [rad])
  };
  struct Result
  {
    bool ok = false;
    Pose pose;
    double matched = 0;      // share of the refinement points matched by the pose
    double second = 0;       // the same for the best distinct other pose
    double search_ms = 0, refine_ms = 0;
    long nodes = 0;          // search nodes expanded
    std::string reason;      // why it failed
  };

  explicit Relocalizer(const VoxelMap &map) : map_(map) { find_floor(); build_pyramid(); }
  Relocalizer(const VoxelMap &map, const Params &params) : map_(map), p_(params) { find_floor(); build_pyramid(); }
  bool has_floor() const { return has_floor_; }
  double floor_z() const { return floor_z_; }

  // The IMU's height above the floor from a submap in L: the densest layer of points below the IMU, refined to
  // the mean of its points. False without a clear floor.
  static bool height_above_floor(const std::vector<V3D> &submap, double &height)
  {
    const double lo = -FLOOR_DEPTH_MAX, hi = -FLOOR_DEPTH_MIN, bin = FLOOR_BIN, r2 = FLOOR_RADIUS * FLOOR_RADIUS;
    std::vector<int> hist(int((hi - lo) / bin) + 1, 0);
    for (const V3D &p : submap)
      if (p(2) > lo && p(2) < hi && p.head<2>().squaredNorm() < r2)
        hist[int((p(2) - lo) / bin)]++;
    const int best = std::max_element(hist.begin(), hist.end()) - hist.begin();
    if (hist[best] < FLOOR_MIN_POINTS)
      return false;
    const double centre = lo + (best + 0.5) * bin;
    double sum = 0;
    int n = 0;
    for (const V3D &p : submap)
      if (std::fabs(p(2) - centre) < FLOOR_REFINE && p.head<2>().squaredNorm() < r2)
        sum += p(2), n++;
    height = -sum / n;
    return true;
  }

  // The pose of L on the map near guess (its z is ignored), or anywhere on the map. height: the IMU's above the
  // floor (NaN: unknown, the mapping sensor's height is used).
  Result localize(const std::vector<V3D> &submap, double height, const Pose &guess) const
  {
    return search(submap, height, &guess);
  }
  Result localize_global(const std::vector<V3D> &submap, double height) const
  {
    return search(submap, height, nullptr);
  }

  private:
  Result search(const std::vector<V3D> &submap, double height, const Pose *guess) const
  {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    Result res;
    std::vector<V3D> in_range, structure;
    for (const V3D &p : submap)
      if (p.squaredNorm() < p_.max_range * p_.max_range)
        in_range.push_back(p);
    if (in_range.size() < MIN_POINTS || pyramid_.empty())
    {
      res.reason = pyramid_.empty() ? "empty map" : "too few submap points";
      return res;
    }
    const bool band = std::isfinite(height) && p_.z_window <= 0;   // the floor, so the band above it, is trusted
    for (const V3D &p : in_range)
      if (!band || (p(2) + height > p_.band_low && p(2) + height < p_.band_high))
        structure.push_back(p);
    const std::vector<V3D> search_pts =
        subsample(structure.size() >= MIN_POINTS ? structure : in_range, p_.search_points);
    const std::vector<V3D> z_pts = subsample(in_range, p_.search_points);
    const std::vector<V3D> refine_pts = subsample(in_range, p_.refine_points);
    const double z0 = has_floor_ && std::isfinite(height) ? floor_z_ + height : 0.0;
    const double s = map_.params().size, dyaw = s / p_.max_range;
    const int n_yaw = int(std::ceil(2 * M_PI / dyaw));

    // the yaws and the translations (voxels) searched: around the guess, or all of them over the map's box
    std::vector<int> yaws;
    Eigen::Vector2i lo(box_min_(0), box_min_(1)), hi(box_max_(0), box_max_(1));
    if (guess)
    {
      const int c = int(std::lround(guess->yaw / (2 * M_PI) * n_yaw)), w = int(std::ceil(p_.yaw_window * M_PI / 180 / dyaw));
      for (int a = c - w; a <= c + w && int(yaws.size()) < n_yaw; a++)
        yaws.push_back(((a % n_yaw) + n_yaw) % n_yaw);
      lo = Eigen::Vector2i(int(std::floor((guess->x - p_.xy_window) / s)), int(std::floor((guess->y - p_.xy_window) / s)));
      hi = Eigen::Vector2i(int(std::floor((guess->x + p_.xy_window) / s)), int(std::floor((guess->y + p_.xy_window) / s)));
    }
    else
      for (int a = 0; a < n_yaw; a++)
        yaws.push_back(a);
    int K = 0;
    while ((1 << K) < (hi - lo).maxCoeff() + 1 && K + 1 < int(pyramid_.size()))
      K++;

    // per yaw: each point's voxel index for the zero translation (a translation of j voxels moves it by j)
    std::vector<std::vector<Eigen::Vector3i>> base(yaws.size(), std::vector<Eigen::Vector3i>(search_pts.size()));
    for (size_t a = 0; a < yaws.size(); a++)
    {
      const double yaw = 2 * M_PI * yaws[a] / n_yaw, c = std::cos(yaw), sn = std::sin(yaw);
      for (size_t i = 0; i < search_pts.size(); i++)
      {
        const V3D &p = search_pts[i];
        base[a][i] = Eigen::Vector3i(int(std::floor((c * p(0) - sn * p(1)) / s)), int(std::floor((sn * p(0) + c * p(1)) / s)),
                                     int(std::floor((p(2) + z0) / s)));
      }
    }
    struct Node
    {
      int bound, level, yaw, ix, iy;
      bool operator<(const Node &o) const { return bound < o.bound || (bound == o.bound && level > o.level); }
    };
    const auto bound = [&](int level, int a, int ix, int iy) {
      int n = 0;
      for (const Eigen::Vector3i &b : base[a])
        n += value_at(level, b(0) + ix, b(1) + iy, b(2));
      return n;
    };
    std::priority_queue<Node> queue;
    for (int a = 0; a < int(yaws.size()); a++)
      for (int ix = lo(0); ix <= hi(0); ix += 1 << K)
        for (int iy = lo(1); iy <= hi(1); iy += 1 << K)
          queue.push({bound(K, a, ix, iy), K, a, ix, iy});
    std::vector<Pose> starts;
    int best_score = 0;
    while (!queue.empty() && int(starts.size()) < p_.candidates)
    {
      const Node n = queue.top();
      queue.pop();
      if (n.bound < p_.min_share * best_score)
        break;   // nothing left can come close to the best
      if (n.level == 0)
      {
        const Pose pose{n.ix * s, n.iy * s, z0, std::remainder(2 * M_PI * yaws[n.yaw] / n_yaw, 2 * M_PI)};
        bool distinct = true;
        for (const Pose &st : starts)
          distinct = distinct && !close(pose, st, DISTINCT_XY, DISTINCT_YAW_DEG * M_PI / 180);
        if (distinct)
        {
          starts.push_back(pose);
          best_score = std::max(best_score, n.bound);
        }
        continue;
      }
      res.nodes++;
      const int h = 1 << (n.level - 1);
      for (int dx = 0; dx < 2; dx++)
        for (int dy = 0; dy < 2; dy++)
          queue.push({bound(n.level - 1, n.yaw, n.ix + dx * h, n.iy + dy * h), n.level - 1, n.yaw, n.ix + dx * h,
                      n.iy + dy * h});
    }
    res.search_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    if (starts.empty())
    {
      res.reason = "no candidate";
      return res;
    }
    finish(z_pts, refine_pts, z0, starts, res);
    return res;
  }

  // z (z_window; z_pts: all heights), refinement and acceptance of the search's best distinct poses.
  void finish(const std::vector<V3D> &z_pts, const std::vector<V3D> &refine_pts, double z0,
              const std::vector<Pose> &starts, Result &res) const
  {
    using clock = std::chrono::steady_clock;
    const auto t1 = clock::now();
    const double deg = M_PI / 180;
    std::vector<std::pair<double, Pose>> refined;
    for (Pose pose : starts)
    {
      const double c = std::cos(pose.yaw), s = std::sin(pose.yaw);
      int best = -1;
      for (double dz = -p_.z_window; p_.z_window > 0 && dz <= p_.z_window + 1e-9; dz += p_.z_step)
      {
        int score = 0;
        for (const V3D &p : z_pts)
          score += map_.occupied(V3D(c * p(0) - s * p(1) + pose.x, s * p(0) + c * p(1) + pose.y, p(2) + z0 + dz),
                                OCCUPIED_POINTS);
        if (score > best || (score == best && std::fabs(dz) < std::fabs(pose.z - z0)))   // ties: nearest the start
          best = score, pose.z = z0 + dz;
      }
      const double m = refine(refine_pts, pose);
      refined.push_back({m, pose});
    }
    std::sort(refined.begin(), refined.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    res.refine_ms = std::chrono::duration<double, std::milli>(clock::now() - t1).count();

    /*** 3. acceptance ***/
    res.pose = refined[0].second;
    res.matched = refined[0].first;
    for (size_t i = 1; i < refined.size(); i++)
      if (!close(refined[i].second, res.pose, OTHER_XY, OTHER_YAW_DEG * deg))
      {
        res.second = refined[i].first;
        break;
      }
    if (res.matched < p_.accept_matched)
      res.reason = "too few matched points";
    else if (res.second > p_.max_ambiguity * res.matched)
      res.reason = "ambiguous";
    else
      res.ok = true;
  }

  public:
  // Gauss-Newton on x, y, z, yaw with point-to-distribution residuals; returns the matched share.
  double refine(const std::vector<V3D> &pts, Pose &pose, int iterations = REFINE_ITERATIONS) const
  {
    const M3D meas = POINT_COV * M3D::Identity();
    int matched = 0;
    for (int it = 0; it <= iterations; it++)
    {
      const double c = std::cos(pose.yaw), s = std::sin(pose.yaw);
      Eigen::Matrix4d H = Eigen::Matrix4d::Zero();
      Eigen::Vector4d b = Eigen::Vector4d::Zero();
      matched = 0;
      for (const V3D &p : pts)
      {
        const V3D q(c * p(0) - s * p(1) + pose.x, s * p(0) + c * p(1) + pose.y, p(2) + pose.z);
        VoxelMap::Distribution dist;
        double d2;
        if (!map_.distribution_match(q, meas, dist, d2) || d2 > GATE)
          continue;
        const V3D &mean = dist.mean;
        matched++;
        const M3D A = inverse_cholesky3(dist.cov + meas);
        Eigen::Matrix<double, 3, 4> J;
        J.leftCols<3>() = A;
        J.col(3) = A * V3D(-s * p(0) - c * p(1), c * p(0) - s * p(1), 0);
        const V3D e = A * (q - mean);
        H += J.transpose() * J;
        b += J.transpose() * e;
      }
      if (it == iterations || matched < REFINE_MIN_MATCHED)
        break;
      const Eigen::Vector4d dx = -H.ldlt().solve(b);
      pose.x += dx(0), pose.y += dx(1), pose.z += dx(2), pose.yaw += dx(3);
      if (dx.head<3>().norm() < CONVERGED_XYZ && std::fabs(dx(3)) < CONVERGED_YAW)
        iterations = it + 1;   // one more pass to count the matches at the final pose
    }
    return double(matched) / pts.size();
  }

  private:
  static std::vector<V3D> subsample(const std::vector<V3D> &pts, int n)
  {
    if (int(pts.size()) <= n)
      return pts;
    std::vector<V3D> out;
    const double stride = double(pts.size()) / n;
    for (int i = 0; i < n; i++)
      out.push_back(pts[size_t(i * stride)]);
    return out;
  }

  static bool close(const Pose &a, const Pose &b, double dist, double angle)
  {
    const double dyaw = std::remainder(a.yaw - b.yaw, 2 * M_PI);
    return std::hypot(a.x - b.x, a.y - b.y) < dist && std::fabs(dyaw) < angle;
  }

  // The score pyramid of the search, dense over the map's voxel box: level 0 the voxel values (OCCUPIED: at least
  // OCCUPIED_POINTS points, NEIGHBOUR: one of its 26 neighbours has), level k their maximum over the 2^k x 2^k block
  // (x, y) starting at the voxel; up to the box's size.
  void build_pyramid()
  {
    bool any = false;
    map_.for_each([&](const VoxelMap::Info &v) {
      if (v.n < OCCUPIED_POINTS)
        return;
      box_min_ = any ? box_min_.cwiseMin(v.index) : v.index;
      box_max_ = any ? box_max_.cwiseMax(v.index) : v.index;
      any = true;
    });
    if (!any)
      return;
    box_min_ -= Eigen::Vector3i::Ones();   // room for the neighbours' value
    box_max_ += Eigen::Vector3i::Ones();
    dims_ = box_max_ - box_min_ + Eigen::Vector3i::Ones();
    std::vector<uint8_t> level(size_t(dims_(0)) * dims_(1) * dims_(2), 0);
    map_.for_each([&](const VoxelMap::Info &v) {
      if (v.n < OCCUPIED_POINTS)
        return;
      for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
          for (int z = -1; z <= 1; z++)
          {
            uint8_t &c = level[cell(v.index(0) + x, v.index(1) + y, v.index(2) + z)];
            c = std::max(c, NEIGHBOUR);
          }
    });
    map_.for_each([&](const VoxelMap::Info &v) {
      if (v.n >= OCCUPIED_POINTS)
        level[cell(v.index(0), v.index(1), v.index(2))] = OCCUPIED;
    });
    pyramid_.push_back(level);
    for (int k = 1; (1 << (k - 1)) < std::max(dims_(0), dims_(1)) && k <= MAX_LEVEL; k++)
    {
      const std::vector<uint8_t> &prev = pyramid_.back();
      std::vector<uint8_t> next(prev.size(), 0);
      const int h = 1 << (k - 1);
      for (int z = 0; z < dims_(2); z++)
        for (int y = 0; y < dims_(1); y++)
          for (int x = 0; x < dims_(0); x++)
          {
            const size_t i = x + size_t(dims_(0)) * (y + size_t(dims_(1)) * z);
            uint8_t m = prev[i];
            if (x + h < dims_(0))
              m = std::max(m, prev[i + h]);
            if (y + h < dims_(1))
              m = std::max(m, prev[i + size_t(h) * dims_(0)]);
            if (x + h < dims_(0) && y + h < dims_(1))
              m = std::max(m, prev[i + h + size_t(h) * dims_(0)]);
            next[i] = m;
          }
      pyramid_.push_back(std::move(next));
    }
  }

  size_t cell(int ix, int iy, int iz) const
  {
    return (ix - box_min_(0)) + size_t(dims_(0)) * ((iy - box_min_(1)) + size_t(dims_(1)) * (iz - box_min_(2)));
  }

  // The level's value of the block starting at (ix, iy, iz). A block that starts before the box but reaches into
  // it takes the value of the block at the box's edge (it contains the part inside: still an upper bound).
  int value_at(int level, int ix, int iy, int iz) const
  {
    const int w = (1 << level) - 1;
    if (iz < box_min_(2) || iz > box_max_(2) || ix + w < box_min_(0) || ix > box_max_(0) || iy + w < box_min_(1) ||
        iy > box_max_(1))
      return 0;
    return pyramid_[level][cell(std::max(ix, box_min_(0)), std::max(iy, box_min_(1)), iz)];
  }

  // The floor: horizontal planar voxels below the map's origin (the mapping sensor), the lowest height at which
  // they hold at least MAP_FLOOR_SHARE of the weight of the strongest such height (a table top is not the floor),
  // refined to their weighted mean. Ceilings are above the sensor, whatever its mounting.
  void find_floor()
  {
    std::vector<std::pair<double, double>> horizontal;   // mean z, points
    map_.for_each([&](const VoxelMap::Info &v) {
      if (v.valid && v.mean(2) < -MAP_FLOOR_MARGIN &&
          std::fabs(v.normal(2)) > std::cos(MAP_FLOOR_TILT_DEG * M_PI / 180))
        horizontal.push_back({v.mean(2), double(v.n)});
    });
    if (horizontal.size() < MAP_FLOOR_MIN_VOXELS)
      return;
    double lo = horizontal[0].first, hi = lo;
    for (const auto &h : horizontal)
      lo = std::min(lo, h.first), hi = std::max(hi, h.first);
    const double bin = MAP_FLOOR_BIN;
    std::vector<double> hist(int((hi - lo) / bin) + 1, 0);
    for (const auto &h : horizontal)
      hist[int((h.first - lo) / bin)] += h.second;
    const double peak = *std::max_element(hist.begin(), hist.end());
    int floor_bin = 0;
    while (hist[floor_bin] < MAP_FLOOR_SHARE * peak)
      floor_bin++;
    double sum = 0, w = 0;
    for (const auto &h : horizontal)
      if (std::fabs(h.first - (lo + (floor_bin + 0.5) * bin)) < MAP_FLOOR_REFINE)
        sum += h.second * h.first, w += h.second;
    floor_z_ = sum / w;
    has_floor_ = true;
  }

  const VoxelMap &map_;
  Params p_;
  bool has_floor_ = false;
  double floor_z_ = 0;
  std::vector<std::vector<uint8_t>> pyramid_;   // the search's score pyramid
  Eigen::Vector3i box_min_ = Eigen::Vector3i::Zero(), box_max_ = Eigen::Vector3i::Zero(), dims_ = Eigen::Vector3i::Zero();
};
