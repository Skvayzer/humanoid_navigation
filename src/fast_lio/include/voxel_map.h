#pragma once
// The voxel map (map_type: voxel), the alternative to the ikd-Tree.
//
// The world is cut into cubic voxels of `size`. Every registered scan point is added to the weighted
// statistics of its voxel (count, weight, weighted sum and sum of outer products), so no observation is
// discarded and no point is chosen by its position in the grid. The weight is 1/r^2 (range r from the
// sensor, at least 1 m): far points, seen at grazing angles, carry range-dependent offsets of centimetres
// to decimetres, so near observations must dominate once they exist. The weights say which observations
// the average prefers, not how precise it is: a point is as precise as the sensor, point_sigma, at any
// range, and the average of a voxel is as precise as point_sigma / sqrt(n_eff), with the effective sample
// size n_eff = (sum w)^2 / sum w^2 (the count for points of equal weight; when near points dominate a mix,
// about their number).
//
// Nothing depends on a number of points. A voxel's distribution is its weighted mean and the covariance
// Σ = S + σ² / n_eff I: the spread S of its points (the surface) plus the uncertainty of their mean. It is
// used on its own once that spread is resolved above the uncertainty of the mean in two directions
// (shape_ratio; one resolved direction is a line of points along one scan line) and the voxel has min_n_eff
// effective points (a covariance of fewer points has the shape of the sample); until then the surface is
// estimated from the neighbourhood instead, the pooled statistics of the 27 or 125 voxels around the point
// (sums are additive), a coarser surface estimate over a larger patch. So a young or sparsely observed region
// is matched like FAST-LIO's neighbourhood plane fit, without stored points or a fixed neighbour count, and a
// well-observed voxel on its own. A voxel freezes once its position is known to freeze_sigma, i.e.
// n_eff >= (point_sigma / freeze_sigma)^2: a map that keeps adapting to the current pose drifts with it. A
// sparser scan gives fewer, coarser measurements, a denser one more and finer ones; the rules stay the same.
//
// Matching (laserMapping.cpp, h_share_model): a measurement takes the distribution among its voxel and the
// 6 face neighbours that it most likely belongs to (distribution_match), the pooled neighbourhood if none
// of them is resolved.
// Moving objects (free_space, on by default): rays from the sensor to a random subset of each scan's points are
// cast through the map (carve, the rays walked in parallel). A ray that passes through a voxel's distribution and
// ends well beyond it saw through it; a scan point that fits a voxel's distribution confirms it (insert). Each voxel
// is removed after `remove_after` scans in a row that see through it without confirming it, frozen or not: what a
// person left behind. Space that is seen through again and again but also hit (thin poles, mesh, edges, grazing
// views) keeps being confirmed. The same rays and the scan's points keep evidence of free space in cells,
// `subdivision` per voxel edge (free_cells.h; 0: none): points that fall where the rays found free space (a person
// walking) are kept out of the map (observe).
// The map never deletes voxels otherwise (about 230 B each): fine indoors, long outdoor runs would need trimming.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Eigenvalues>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <common_lib.h>
#include <free_cells.h>

class VoxelMap
{
  public:
  struct Params
  {
    double size = 0.35;          // voxel edge [m]
    double point_sigma = 0.02;   // noise of a point [m], at any range (the sensor's precision)
    double shape_ratio = 1.0;    // a voxel is matched on its own once the spread of its points is resolved in two
                                 //   directions (the middle eigenvalue of S exceeds this ratio of the uncertainty of
                                 //   their mean, σ² / n_eff: a surface, not a line of points along one scan line);
                                 //   until then the pooled neighbourhood is matched instead. 0: always on its own
    double min_n_eff = 6;        // ... and once its effective sample size reaches this: a covariance of fewer points
                                 //   has the shape of the sample (2 points a line, 3 a plane), not of the surface
    double freeze_sigma = 0.004; // a voxel takes no more points once its position is known to this [m], i.e.
                                 //   n_eff >= (point_sigma / freeze_sigma)^2; 0: never
    double pooled_thickness = 0.1; // a pooled neighbourhood is one surface if its points scatter no more than this
    double pooled_planarity = 0.2; //   [m] (std. deviation) about their plane, and at most this ratio of the plane's
                                   //   smaller extent (FAST-LIO's neighbourhood fit accepted a 10 cm band)

    // The parameters that differ from o's: "name this / o, ...", "" if none.
    std::string differences(const Params &o) const
    {
      std::ostringstream out;
      const auto cmp = [&out](const char *name, double a, double b) {
        if (a != b)
          out << (out.tellp() > 0 ? ", " : "") << name << " " << a << " / " << b;
      };
      cmp("size", size, o.size), cmp("point_sigma", point_sigma, o.point_sigma);
      cmp("shape_ratio", shape_ratio, o.shape_ratio), cmp("min_n_eff", min_n_eff, o.min_n_eff);
      cmp("freeze_sigma", freeze_sigma, o.freeze_sigma);
      cmp("pooled_thickness", pooled_thickness, o.pooled_thickness), cmp("pooled_planarity", pooled_planarity, o.pooled_planarity);
      return out.str();
    }
  };
  // Removal of what moved (carve, insert, settle). Not part of the map file.
  struct FreeSpace
  {
    bool enable = true;
    double ray_share = 0.15;    // rays cast per scan: this share of its points, a random subset (a denser lidar gets
                                //   more rays; 0.15 of a thinned Mid-360 scan is about 1000)
    double min_range = 0.5;     // a ray starts this far from the sensor (the robot itself)
    double max_range = 8.0;     // ... and is followed this far at most [m]: farther, the pose's angular error and the
                                //   beam width make a ray's path too uncertain to tell through from beside
    double ray_sigma = 0.03;    // uncertainty of a ray's end point and of a scan point [m] (range noise, pose), added
                                //   to a distribution's covariance in the `beyond` and `confirm` tests (not in `core`:
                                //   there it would let a ray that passes beside a small surface's edge count as
                                //   passing through it)
    double core = 2.0;          // a ray passes through a distribution if it comes within this Mahalanobis distance
                                //   of its mean
    double beyond = 3.0;        // ... and ends past it, farther than this Mahalanobis distance from its mean
    double confirm = 3.0;       // a scan point within this Mahalanobis distance of a distribution confirms it
    int remove_after = 5;       // a voxel is removed after this many scans in a row that see through it without a
                                //   confirming point (a confirmation starts the count again)
    int threads = 1;            // OpenMP threads walking the rays (the node: num_threads)
    FreeCells::Params cell;     // free space in cells (free_cells.h; cell.subdivision 0: none): points that fall
                                //   where the rays found free space are kept out of the map (observe)
    // grazing rays (a floor or a low ceiling seen at a shallow angle) would mark the cells of the surface itself free:
    double graze_k = 3.0;       // a ray marks cells free only up to where it enters the band of +-graze_k sigma
                                //   about the surface it ends on (its end voxel's plane, own or pooled): its margin
                                //   is band / sin(incidence), at least cell.margin (0: cell.margin only)
    double band_k = 2.0;        // and no ray marks free a cell within +-band_k sigma of its own voxel's plane (0: off)
  };
  static constexpr const char *EXTENSION = ".vmap";

  // A distribution: mean, covariance Σ = S + var_mean I (the spread of the points plus the uncertainty of their
  // mean, var_mean = σ² / n_eff) and the effective sample size.
  struct Distribution
  {
    V3D mean = V3D::Zero();
    M3D cov = M3D::Zero();
    double n_eff = 0, var_mean = 0;
  };

  // The weight of a point at range r: 1 / r² from 1 m (near observations dominate the average).
  static double weight(double r) { return 1.0 / (std::max(1.0, r) * std::max(1.0, r)); }
  // The noise variance of a point.
  double noise2() const { return p_.point_sigma * p_.point_sigma; }

  void set_params(const Params &params) { p_ = params; }
  void set_free_space(const FreeSpace &fs)
  {
    fs_ = fs;
    cells_.set(p_.size, fs.cell);
  }
  // Free space (cells, subdivision > 0): marks a scan point at world point p as a hit and tells whether it lies where the
  // previous scans found free space (to keep out of the map). Before the scan's settle().
  bool observe(const V3D &p)
  {
    if (!fs_.enable || !cells_on())
      return false;
    const bool free = cells_.is_free(p);
    cells_.hit(p);
    return free;
  }
  size_t free_cell_blocks() const { return cells_.blocks(); }
  bool cells_on() const { return fs_.cell.subdivision > 0; }
  const FreeSpace &free_space() const { return fs_; }
  // Keys (pack) of the voxels the last settle() removed.
  const std::vector<uint64_t> &removed() const { return removed_; }
  const Params &params() const { return p_; }
  bool empty() const { return voxels_.empty(); }
  size_t size() const { return voxels_.size(); }

  // Adds points (world frame), measured at the given ranges from the sensor. Returns the number of new voxels.
  int insert(const PointVector &points, const std::vector<float> &ranges)
  {
    int created = 0;
    touched_.clear();
    for (size_t i = 0; i < points.size(); i++)
    {
      const PointType &pt = points[i];
      const V3D p(pt.x, pt.y, pt.z);
      const double w = weight(ranges[i]);
      const Eigen::Vector3i idx = index_of(p);
      Voxel &v = voxels_[pack(idx)];
      // confirmation, against the voxel as the scan found it (its test record is renewed after the insertion)
      if (fs_.enable && v.n > 0 && v.born != scan_ && !v.confirmed && (v.seen_through || v.misses > 0) &&
          test(v).mahalanobis2(p) <= fs_.confirm * fs_.confirm)
      {
        v.confirmed = true;
        if (!v.seen_through)
          pending_.push_back(&v);
      }
      if (frozen(v))
        continue;  // frozen: a map that keeps adapting to the current pose drifts with it
      if (v.n == 0)
      {
        v.origin = idx.cast<double>() * p_.size;  // statistics are kept relative to the voxel corner
        v.key = pack(idx);
        v.misses = 0;
        v.born = scan_;
        created++;
        mark(idx, true);
      }
      const V3D q = p - v.origin;
      v.n++;
      v.w += w;
      v.w2 += w * w;
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
      refresh(*v);
      v->touched = false;
      v->cached = false;
    }
    return created;
  }

  // Ray casting (free_space.enable), before the scan's insert(): rays from the sensor (pose R, origin in the world)
  // to a random subset of the scan points (ray_share; sensor frame), each from min_range to its end point or
  // max_range. The subset is a hash of the point index seeded per scan: an even stride would follow the order the
  // lidar's lasers fire in (every 4th point of a Mid-360 is the same laser) and pick one laser's points. A ray sees
  // through a voxel if it passes within `core` of its distribution (the closest approach of the ray's line, in
  // the distribution's Mahalanobis metric) before its end point, and the end point lies past it: for a distribution
  // that is one surface, on the surface's far side by more than `beyond` times the spread across it (with
  // ray_sigma), otherwise farther than `beyond` from it (Mahalanobis, with ray_sigma). Such voxels become suspects.
  void carve(const M3D &R, const V3D &origin, const PointVector &scan)
  {
    if (!fs_.enable || voxels_.empty() || scan.empty())
      return;
    const double share = std::min(1.0, fs_.ray_share);
    const uint64_t seed = splitmix64(scan_);
    ray_ends_.clear();
    // hash * 2^-53 < share, exact in double as in Preprocess::keep
    const uint64_t threshold = uint64_t(std::ceil(share * 9007199254740992.0));
    for (size_t i = 0; i < scan.size(); i++)
      if ((splitmix64(seed ^ splitmix64(i)) >> 11) < threshold)
        ray_ends_.push_back(R * V3D(scan[i].x, scan[i].y, scan[i].z) + origin);
    const int threads = std::max(1, fs_.threads);
    // grazing (graze_k): per ray, how far before its end it stops marking cells free (-1: cell.margin), from the
    // surface of the voxel it ends in (their tests made ready in parallel)
    ray_margin_.assign(ray_ends_.size(), -1.0);
    if (cells_on() && fs_.graze_k > 0)
    {
      ray_voxel_.assign(ray_ends_.size(), nullptr);
#pragma omp parallel for num_threads(threads) schedule(static)
      for (int k = 0; k < int(ray_ends_.size()); k++)
      {
        const double r = (ray_ends_[k] - origin).norm();
        if (r > fs_.min_range && r <= fs_.max_range)
          ray_voxel_[k] = find(index_of(ray_ends_[k]));
      }
      prepare(ray_voxel_, SURFACE, threads);
#pragma omp parallel for num_threads(threads) schedule(static)
      for (int k = 0; k < int(ray_ends_.size()); k++)
      {
        if (!ray_voxel_[k])
          continue;
        const Test &t = tests_[ray_voxel_[k]->test];
        if (!t.planar && !t.pooled_surface)
          continue;
        const V3D ray = ray_ends_[k] - origin;
        const double r = ray.norm();
        const double s = std::abs((t.planar ? t.normal : t.pooled_normal).cast<double>().dot(ray / r));
        const double band = fs_.graze_k * std::sqrt(t.planar ? t.across2 : t.pooled_across2);
        ray_margin_[k] = std::min(r, std::max(fs_.cell.margin, band / std::max(s, 1e-3)));
      }
    }
    // 1. Walking the rays, in parallel and reading only: the stored voxels each passes and the cells it crosses
    //    (with the stored voxel of each block of cells, for the band). Each thread takes a contiguous run of rays
    //    (static schedule), so its lists, concatenated, are in ray order.
    if (int(walks_.size()) < threads)
      walks_.resize(threads);
    const bool band = cells_on() && fs_.band_k > 0;
#pragma omp parallel num_threads(threads)
    {
#ifdef _OPENMP
      Walk &w = walks_[omp_get_thread_num()];
#else
      Walk &w = walks_[0];
#endif
      w.passed.clear();
      w.cells.clear();
      w.cell_voxels.clear();
#pragma omp for schedule(static)
      for (int k = 0; k < int(ray_ends_.size()); k++)
      {
        const V3D ray = ray_ends_[k] - origin;
        const double r = ray.norm();
        if (r <= fs_.min_range)
          continue;
        const V3D d = ray / r;
        traverse(origin, d, fs_.min_range, std::min(r, fs_.max_range),
                 [&](Voxel &v) { w.passed.emplace_back(&v, k); });
        if (cells_on())
          cells_.walk(origin, d, fs_.min_range, r > fs_.max_range ? fs_.max_range + fs_.cell.margin : r, w.cells,
                      ray_margin_[k]);
        if (band)
          for (size_t i = w.cell_voxels.size(); i < w.cells.size(); i++)
            w.cell_voxels.push_back(find(unpack(w.cells[i].key)));
      }
    }
    // 2. The tests (made ready in parallel; applied in ray order, they cache per voxel) and the cells' misses
    prepared_list_.clear();
    for (int t = 0; t < threads; t++)
      for (const auto &pv : walks_[t].passed)
        prepared_list_.push_back(pv.first);
    prepare(prepared_list_, TEST, threads);
    if (band)
    {
      prepared_list_.clear();
      for (int t = 0; t < threads; t++)
        prepared_list_.insert(prepared_list_.end(), walks_[t].cell_voxels.begin(), walks_[t].cell_voxels.end());
      prepare(prepared_list_, BAND, threads);
    }
    for (int t = 0; t < threads; t++)
    {
      for (const auto &pv : walks_[t].passed)
      {
        Voxel &v = *pv.first;
        if (v.n == 0 || v.missed == scan_)
          continue;
        const V3D &end = ray_ends_[pv.second];
        if (sees_through(v, origin, end))
        {
          v.missed = scan_;
          v.seen_through = true;
          pending_.push_back(&v);
        }
      }
      if (band)
      {
        Walk &w = walks_[t];
        for (size_t i = 0; i < w.cells.size(); i++)
          if (w.cell_voxels[i])
            w.cells[i].cells &= ~band_mask(*w.cell_voxels[i]);
      }
      if (cells_on())
        cells_.add_misses(walks_[t].cells);
    }
  }

  // After the scan's insert(): a voxel confirmed by the scan starts its count again, one seen through without a
  // confirmation counts one more and is removed at remove_after. Returns the number removed (keys in removed()).
  int settle()
  {
    removed_.clear();
    if (!fs_.enable)
      return 0;
    for (Voxel *v : pending_)
    {
      if (v->confirmed)
        v->misses = 0;
      else
        v->misses++;   // seen through, no point fits it
      v->confirmed = v->seen_through = false;
      if (v->misses >= fs_.remove_after)
        removed_.push_back(v->key);
    }
    pending_.clear();
    if (cells_on())
      cells_.update(std::max(1, fs_.threads));
    for (uint64_t key : removed_)
    {
      const auto it = voxels_.find(key);
      if (it->second.test >= 0)
        free_tests_.push_back(it->second.test);
      voxels_.erase(it);   // erasing keeps the other elements' pointers valid
      mark(unpack(key), false);
    }
    scan_++;
    return int(removed_.size());
  }

  // The distribution that a measurement at world point p with covariance meas_cov most likely belongs to,
  // among the resolved ones of its voxel and the 6 face neighbours: the smallest d2 + log det (cov + meas_cov),
  // d2 the squared Mahalanobis distance r' (cov + meas_cov)^-1 r, r = p - mean (the Gaussian log-likelihood:
  // a vague distribution does not win by being vague). If none of the 7 is resolved, the pooled distribution
  // of the voxels around p's voxel (pooled = true; blob = true if it is no single surface). d2 is that of the
  // chosen one. False if nothing resolved is there.
  bool distribution_match(const V3D &p, const M3D &meas_cov, Distribution &out, double &d2,
                          bool *pooled = nullptr, bool *blob = nullptr) const
  {
    static const int offsets[7][3] = {{0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const Eigen::Vector3i idx = index_of(p);
    bool found = false;
    double best = std::numeric_limits<double>::max();
    if (pooled != nullptr)
      *pooled = false;
    if (blob != nullptr)
      *blob = false;
    for (const auto &o : offsets)
    {
      const auto it = voxels_.find(pack(idx + Eigen::Vector3i(o[0], o[1], o[2])));
      if (it == voxels_.end() || !it->second.resolved)
        continue;
      const Voxel &v = it->second;
      Distribution d;
      distribution(v.origin, v.w, v.w2, v.sum, v.sum_sq, d);
      double e, score;
      likelihood(p, d, meas_cov, e, score);
      if (score < best)
      {
        best = score;
        d2 = e;
        out = d;
        found = true;
      }
    }
    if (found)
      return true;
    Distribution d;
    bool is_blob = false;
    if (!pooled_distribution(idx, d, is_blob))
      return false;
    double score;
    likelihood(p, d, meas_cov, d2, score);
    out = d;
    if (pooled != nullptr)
      *pooled = true;
    if (blob != nullptr)
      *blob = is_blob;
    return true;
  }

  // The surface around voxel idx from pooled statistics (sums are additive): the 27 voxels around it, or the
  // 125 two rings out if those give nothing usable (sparse scenes: far returns outdoors). Usable means at least
  // 4 effective points (a plane needs 3, a fourth checks it), resolved, and one surface (pooled_thickness,
  // pooled_planarity): a region that is not (a corner, clutter, a corridor's end seen with its walls) has a mean
  // that means nothing as a surface; it is still used as a blob as wide as the region (a weak constraint) when
  // no scale gives a surface. False if nothing resolved is there.
  bool pooled_distribution(const Eigen::Vector3i &idx, Distribution &out, bool &is_blob) const
  {
    Distribution blob;
    bool have_blob = false;
    is_blob = false;
    for (int ring = 1; ring <= 2; ring++)
    {
      Acc a;
      for (int x = -ring; x <= ring; x++)
        for (int y = -ring; y <= ring; y++)
          for (int z = -ring; z <= ring; z++)
          {
            const auto it = voxels_.find(pack(idx + Eigen::Vector3i(x, y, z)));
            if (it != voxels_.end() && it->second.n > 0)
              a.add(it->second);
          }
      if (a.w <= 0)
        continue;
      distribution(a.origin, a.w, a.w2, a.sum, a.sum_sq, out);
      if (!resolved(out, 4.0))
        continue;
      V3D normal;
      double d;
      if (one_surface(out.cov, out.mean, normal, d))
        return true;
      if (!have_blob)
      {
        blob = out;   // not one surface: kept as a weak constraint if no scale gives a surface
        have_blob = true;
      }
    }
    if (have_blob)
      out = blob;
    is_blob = have_blob;
    return have_blob;
  }

  // Whether a distribution is resolved enough to be matched: at least min_n effective points, and the spread
  // of its points resolved in two directions, i.e. the middle eigenvalue of cov (= λ1(S) + σ² / n_eff) exceeds
  // (1 + shape_ratio) σ² / n_eff. One resolved direction only is a line of points (one scan line across the
  // voxel): whether the surface extends across it is unknown, so it is not matched on its own.
  bool resolved(const Distribution &d, double min_n) const
  {
    if (d.n_eff < min_n)
      return false;
    if (p_.shape_ratio <= 0)
      return true;
    const double need = (1.0 + p_.shape_ratio) * d.var_mean;
    if (d.cov.trace() < 2.0 * need)
      return false;   // λ1 + λ2 <= trace
    Eigen::SelfAdjointEigenSolver<M3D> es;
    es.computeDirect(d.cov);
    return es.eigenvalues()(1) >= need;
  }

  // Whether a covariance is one surface (pooled_thickness, pooled_planarity): thin enough about its plane and a
  // surface rather than a line. Gives the plane's normal and offset d (n . x + d = 0).
  bool one_surface(const M3D &cov, const V3D &mean, V3D &normal, double &d) const
  {
    Eigen::SelfAdjointEigenSolver<M3D> es;
    es.computeDirect(cov);
    const V3D l = es.eigenvalues();  // ascending
    normal = es.eigenvectors().col(0);
    d = -normal.dot(mean);
    return l(0) <= p_.pooled_thickness * p_.pooled_thickness && l(0) <= p_.pooled_planarity * l(1);
  }

  // Empties the map, keeping its parameters.
  void clear()
  {
    voxels_.clear();
    pending_.clear();
    blocks_.clear();
    cells_.clear();
    tests_.clear();
    free_tests_.clear();
  }

  // Drops the voxels farther than radius from centre (localization's live layer stays local).
  void trim(const V3D &centre, double radius)
  {
    const double r2 = radius * radius;
    for (auto it = voxels_.begin(); it != voxels_.end();)
    {
      if ((it->second.origin - centre).squaredNorm() <= r2)
      {
        ++it;
        continue;
      }
      if (it->second.test >= 0)
        free_tests_.push_back(it->second.test);
      mark(unpack(it->first), false);
      it = voxels_.erase(it);
    }
    cells_.trim(centre, radius);
  }

  // Whether world point p falls in a voxel with at least min_n points (relocalization's search score).
  bool occupied(const V3D &p, int min_n = 1) const
  {
    const auto it = voxels_.find(pack(index_of(p)));
    return it != voxels_.end() && it->second.n >= min_n;
  }

  // Recomputes every voxel's resolved flag and renews its free-space test record (point_sigma, shape_ratio), after
  // set_params with new thresholds (debug session).
  void refit_all()
  {
    for (auto &kv : voxels_)
    {
      refresh(kv.second);
      kv.second.cached = false;
    }
  }

  // Inspection (debug session, relocalization): one voxel's statistics, in the world frame. normal, d and valid
  // are the plane of the voxel's distribution and whether it is one surface (one_surface).
  struct Info
  {
    Eigen::Vector3i index;
    int n;
    double w, n_eff;
    V3D mean;
    M3D cov;
    V3D normal;
    double d;
    bool valid, frozen, resolved;
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
      Distribution d;
      distribution(v.origin, v.w, v.w2, v.sum, v.sum_sq, d);
      info.n_eff = d.n_eff;
      info.mean = d.mean;
      info.cov = d.cov;
      info.valid = one_surface(d.cov, d.mean, info.normal, info.d);
      info.frozen = frozen(v);
      info.resolved = v.resolved;
      f(info);
    }
  }

  // The map as a file (.vmap, the primary map of a localization run): the pose of the map's frame in the primary
  // map's frame (identity for a SLAM map; a live layer saved in localization mode: map -> init), the parameters
  // and every voxel's statistics. Binary in the machine's byte order (a byte-order mark makes a reader with
  // another one fail cleanly). load() replaces the map and its parameters; a loaded map matches exactly as the
  // saved one did. level (a SLAM map's leveling, see save_map): the voxels are written rotated by it about the
  // origin, each voxel's statistics into the voxel of its rotated centre (the rotated grid maps nearly one to one;
  // voxels meeting there are merged); this map is not changed.
  bool save(const std::string &path, const Eigen::Isometry3d &frame = Eigen::Isometry3d::Identity(),
            const M3D &level = M3D::Identity()) const
  {
    std::unordered_map<uint64_t, Voxel> rotated;
    const std::unordered_map<uint64_t, Voxel> *voxels = &voxels_;
    if (!level.isIdentity(1e-12))
    {
      for (const auto &kv : voxels_)
      {
        const Voxel &v = kv.second;
        if (v.w <= 0)
          continue;
        const V3D o = level * v.origin, s = level * v.sum;
        const Eigen::Vector3i idx = index_of(level * (v.origin + V3D::Constant(p_.size / 2)));
        Voxel &r = rotated[pack(idx)];
        r.origin = idx.cast<double>() * p_.size;
        const V3D d = o - r.origin;   // q' = level q + d: sums relative to the new voxel's corner
        r.n += v.n;
        r.w += v.w;
        r.w2 += v.w2;
        r.sum += s + v.w * d;
        r.sum_sq += level * v.sum_sq * level.transpose() + s * d.transpose() + d * s.transpose() + v.w * d * d.transpose();
      }
      voxels = &rotated;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f)
      return false;
    f.write(MAGIC, 4);
    put(f, VERSION), put(f, BYTE_ORDER_MARK);
    const Eigen::Quaterniond q(frame.rotation());
    put(f, V3D(frame.translation())), put(f, q.x()), put(f, q.y()), put(f, q.z()), put(f, q.w());
    put(f, p_.size), put(f, p_.point_sigma), put(f, p_.shape_ratio), put(f, p_.min_n_eff), put(f, p_.freeze_sigma);
    put(f, p_.pooled_thickness), put(f, p_.pooled_planarity);
    put(f, uint64_t(voxels->size()));
    for (const auto &kv : *voxels)
    {
      const Voxel &v = kv.second;
      put(f, kv.first), put(f, v.n), put(f, v.w), put(f, v.w2), put(f, v.sum), put(f, v.sum_sq);
    }
    return bool(f);
  }

  // Whether the file at path is a .vmap.
  static bool is_vmap(const std::string &path)
  {
    char magic[4] = {0, 0, 0, 0};
    return std::ifstream(path, std::ios::binary).read(magic, 4) && std::string(magic, 4) == std::string(MAGIC, 4);
  }

  // False with the reason in error (if given): not found, a PCD (an ikd-Tree map) or another file, another
  // version or byte order, damaged. frame (if given) gets the saved frame pose.
  bool load(const std::string &path, std::string *error = nullptr, Eigen::Isometry3d *frame = nullptr)
  {
    const auto fail = [error](const std::string &why) {
      if (error)
        *error = why;
      return false;
    };
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
      return fail("not found");
    const std::streamoff file_size = f.tellg();
    f.seekg(0);
    char head[7] = {0, 0, 0, 0, 0, 0, 0};
    uint32_t version = 0, mark = 0;
    f.read(head, 7);
    if (std::string(head, 6) == "# .PCD" || std::string(head, 7) == "VERSION")
      return fail("a PCD (an ikd-Tree map), not a .vmap");
    f.clear();
    f.seekg(4);
    if (std::string(head, 4) != std::string(MAGIC, 4))
      return fail("not a .vmap");
    if (!get(f, version) || version != VERSION)
      return fail("a .vmap of version " + std::to_string(version) + ", this build reads version " +
                  std::to_string(VERSION));
    if (!get(f, mark) || mark != BYTE_ORDER_MARK)
      return fail("written on a machine with another byte order");
    V3D t;
    double qx, qy, qz, qw;
    get(f, t), get(f, qx), get(f, qy), get(f, qz), get(f, qw);
    Params p = p_;
    get(f, p.size), get(f, p.point_sigma), get(f, p.shape_ratio), get(f, p.min_n_eff), get(f, p.freeze_sigma);
    get(f, p.pooled_thickness), get(f, p.pooled_planarity);
    uint64_t n_voxels = 0;
    get(f, n_voxels);
    // a count beyond what the rest of the file can hold: damaged (checked before anything is allocated)
    const uint64_t left = uint64_t(std::max<std::streamoff>(0, file_size - f.tellg()));
    if (!f || n_voxels > left / VOXEL_BYTES)
      return fail("damaged (voxel count beyond the file size)");
    std::unordered_map<uint64_t, Voxel> voxels;
    voxels.reserve(n_voxels);
    for (uint64_t i = 0; i < n_voxels && f; i++)
    {
      uint64_t key;
      Voxel v;
      get(f, key), get(f, v.n), get(f, v.w), get(f, v.w2), get(f, v.sum), get(f, v.sum_sq);
      v.origin = unpack(key).cast<double>() * p.size;   // as insert() sets it
      v.key = key;
      v.misses = 0;   // an established map
      voxels.emplace(key, v);
    }
    if (!f)
      return fail("damaged (truncated)");
    p_ = p;
    voxels_.swap(voxels);
    pending_.clear();
    cells_.clear();
    cells_.set(p_.size, fs_.cell);
    rebuild_blocks();
    tests_.clear();
    free_tests_.clear();
    for (auto &kv : voxels_)
      refresh(kv.second);
    if (frame)
    {
      frame->setIdentity();
      frame->linear() = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
      frame->translation() = t;
    }
    return true;
  }

  // The map as a cloud: per voxel its mean (step 0), or a grid of points `step` apart over its distribution: the
  // box of ±√3 σ along the principal axes (a uniform box with that covariance), cut to the voxel's cube; a voxel that
  // is one surface gets one layer, on its plane. intensity = number of points, normal = plane normal (zero if the
  // voxel is not one surface), curvature = 1 if the voxel is resolved (matched on its own).
  void to_cloud(PointCloudXYZI &out, double step = 0) const
  {
    out.clear();
    const double size = p_.size;
    for_each([&out, step, size](const Info &v) {
      PointType pt;
      pt.intensity = v.n;
      pt.normal_x = v.valid ? v.normal(0) : 0;
      pt.normal_y = v.valid ? v.normal(1) : 0;
      pt.normal_z = v.valid ? v.normal(2) : 0;
      pt.curvature = v.resolved;
      auto add = [&out, &pt](const V3D &p) {
        pt.x = p(0);
        pt.y = p(1);
        pt.z = p(2);
        out.push_back(pt);
      };
      if (step <= 0)
      {
        add(v.mean);
        return;
      }
      Eigen::SelfAdjointEigenSolver<M3D> es;
      es.computeDirect(v.cov);   // ascending: column 0 is the normal of one surface (one_surface)
      V3D half;
      int k[3];
      for (int i = 0; i < 3; i++)
      {
        half(i) = std::sqrt(3 * std::max(es.eigenvalues()(i), 0.0));
        k[i] = (i == 0 && v.valid) ? 1 : std::max(1, int(std::lround(2 * half(i) / step)));
      }
      const V3D lo = v.index.cast<double>() * size, hi = lo + V3D::Constant(size);
      for (int a = 0; a < k[0]; a++)
        for (int b = 0; b < k[1]; b++)
          for (int c = 0; c < k[2]; c++)
          {
            const V3D u((2 * a + 1.0) / k[0] - 1, (2 * b + 1.0) / k[1] - 1, (2 * c + 1.0) / k[2] - 1);
            const V3D p = v.mean + es.eigenvectors() * u.cwiseProduct(half);
            if ((p.array() >= lo.array()).all() && (p.array() <= hi.array()).all())
              add(p);
          }
    });
  }

  private:
  struct Voxel
  {
    V3D origin = V3D::Zero();
    int n = 0;                 // points
    double w = 0, w2 = 0;      // their total weight and sum of squared weights (n_eff = w^2 / w2)
    V3D sum = V3D::Zero();     // weighted, relative to origin
    M3D sum_sq = M3D::Zero();
    bool touched = false;
    bool resolved = false;     // matched on its own (see resolved()), kept up to date by refresh()
    uint64_t key = 0;
    // free space (carve, settle)
    uint32_t missed = 0;       // the last scan (scan_) a ray saw through it
    uint32_t born = 0;         // the scan that created it
    int misses = 0;            // scans in a row that saw through it without a confirmation
    bool seen_through = false, confirmed = false;   // in this scan (then in pending_)
    int32_t test = -1;         // its slot in tests_, -1: none yet
    uint32_t prepared = 0;     // scan * 4 + Readiness of the last prepare() that made its test ready (carve)
    bool cached = false;       // ... and it is up to date
  };

  // A voxel's distribution as the free-space tests use it: mean and L^-1 (C = L L') of C = cov + ray_sigma^2 I (end
  // points, scan points) and Lc of cov itself (a ray's path).
  struct Test
  {
    V3D mean;
    Eigen::Matrix3f L, Lc;
    bool planar;               // one surface (one_surface): normal, and the variance across it with ray_sigma^2
    Eigen::Vector3f normal;
    float across2;
    bool pooled_surface = false;   // else: on the surface of its pooled neighbourhood, with its normal and variance
    Eigen::Vector3f pooled_normal; //   across (on_surface, computed in scan pooled_scan; 0: not yet)
    float pooled_across2;
    uint32_t pooled_scan = 0;
    uint64_t band = 0;             // band_mask, computed in scan band_scan
    uint32_t band_scan = 0;
    float mahalanobis2(const V3D &p) const { return (L * (p - mean).cast<float>()).squaredNorm(); }
  };

  // Statistics added up over voxels (the pooled neighbourhood), relative to the first voxel's origin.
  struct Acc
  {
    V3D origin = V3D::Zero();
    double w = 0, w2 = 0;
    V3D sum = V3D::Zero();
    M3D sum_sq = M3D::Zero();
    void add(const Voxel &v)
    {
      if (w <= 0)
        origin = v.origin;
      const V3D s = v.origin - origin;   // q' = q + s: sums move with the origin
      w += v.w;
      w2 += v.w2;
      sum += v.sum + v.w * s;
      sum_sq += v.sum_sq + v.sum * s.transpose() + s * v.sum.transpose() + v.w * s * s.transpose();
    }
  };

  // The distribution of statistics (total weight w, sum of squared weights w2, sums relative to origin): the mean
  // is uncertain by σ² / n_eff.
  void distribution(const V3D &origin, double w, double w2, const V3D &sum, const M3D &sum_sq, Distribution &d) const
  {
    const V3D m = sum / w;
    d.n_eff = w * w / w2;
    d.var_mean = noise2() / d.n_eff;
    d.cov = sum_sq / w - m * m.transpose() + d.var_mean * M3D::Identity();
    d.mean = origin + m;
  }

  // The squared Mahalanobis distance e of p to d under cov + meas_cov, and the log-likelihood score e + log det.
  static void likelihood(const V3D &p, const Distribution &d, const M3D &meas_cov, double &e, double &score)
  {
    const M3D Linv = inverse_cholesky3(d.cov + meas_cov);   // L^-1, C = L L'
    e = (Linv * (p - d.mean)).squaredNorm();
    score = e - 2.0 * std::log(Linv(0, 0) * Linv(1, 1) * Linv(2, 2));
  }

  // A slot in tests_ for v's test record (to be computed).
  void take_slot(Voxel &v)
  {
    if (free_tests_.empty())
    {
      v.test = int32_t(tests_.size());
      tests_.emplace_back();
    }
    else
    {
      v.test = free_tests_.back();
      free_tests_.pop_back();
    }
    v.cached = false;
  }
  // The voxel's test distribution, cached (in tests_, slot v.test) until a point is added.
  Test &test(Voxel &v)
  {
    if (v.test < 0)
      take_slot(v);
    Test &t = tests_[v.test];
    if (!v.cached)
    {
      Distribution d;
      distribution(v.origin, v.w, v.w2, v.sum, v.sum_sq, d);
      t.mean = d.mean;
      t.L = inverse_cholesky3(d.cov + fs_.ray_sigma * fs_.ray_sigma * M3D::Identity()).cast<float>();
      t.Lc = inverse_cholesky3(d.cov).cast<float>();
      V3D normal;
      double offset;
      t.planar = one_surface(d.cov, d.mean, normal, offset);
      t.normal = normal.cast<float>();
      t.across2 = normal.dot(d.cov * normal) + fs_.ray_sigma * fs_.ray_sigma;
      t.pooled_scan = 0;
      v.cached = true;
    }
    return t;
  }

  // The cells of voxel v within +-band_k sigma of its plane (its own or its neighbourhood's), plus half a cell's
  // extent along the normal: a ray marks none of them free (band_k).
  uint64_t band_mask(Voxel &v)
  {
    if (v.n == 0)
      return 0;
    Test &t = test(v);
    if (!on_surface(v, t))
      return 0;
    if (t.band_scan == scan_)
      return t.band;
    const V3D n = (t.planar ? t.normal : t.pooled_normal).cast<double>();
    const double half = 0.5 * cells_.cell_size() * (std::abs(n(0)) + std::abs(n(1)) + std::abs(n(2)));
    const double band = fs_.band_k * std::sqrt(t.planar ? t.across2 : t.pooled_across2) + half;
    // the cell centres' offsets from the mean, per axis (cell i of the block: i % s, i / s % s, i / s^2; s <= 4)
    const int s = cells_.subdivision();
    const Eigen::Vector3i c0 = unpack(v.key) * s;
    double off[3][4];
    for (int ax = 0; ax < 3; ax++)
      for (int j = 0; j < s; j++)
      {
        const double centre = (double(c0(ax) + j) + 0.5) * cells_.cell_size();
        off[ax][j] = centre - t.mean(ax);
      }
    t.band = 0;
    for (int i = 0; i < s * s * s; i++)
      if (std::abs(n.dot(V3D(off[0][i % s], off[1][i / s % s], off[2][i / (s * s)]))) <= band)
        t.band |= uint64_t(1) << i;
    t.band_scan = scan_;
    return t.band;
  }

  // Makes the test records of voxels (null: skipped) ready for this scan in parallel, as the sequential calls would:
  // TEST the distribution's test, SURFACE also whether it is (on) a surface, BAND also its band mask. Slots are
  // taken first, sequentially; then each voxel's record is its own (its neighbourhood is only read).
  enum Readiness { TEST = 1, SURFACE = 2, BAND = 3 };
  void prepare(const std::vector<Voxel *> &vs, Readiness what, int threads)
  {
    prepared_.clear();
    const uint32_t stamp = scan_ * 4 + what;
    for (Voxel *v : vs)
      if (v && v->n > 0 && v->prepared != stamp)
      {
        v->prepared = stamp;
        if (v->test < 0)
          take_slot(*v);
        prepared_.push_back(v);
      }
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
    for (int i = 0; i < int(prepared_.size()); i++)
    {
      Voxel &v = *prepared_[i];
      Test &t = test(v);
      if (what == BAND)
        band_mask(v);
      else if (what == SURFACE)
        on_surface(v, t);
    }
  }

  // The stored voxel at idx (none: null), reading only: an empty block of the occupancy is skipped without a lookup.
  Voxel *find(const Eigen::Vector3i &idx)
  {
    const auto bt = blocks_.find(pack(Eigen::Vector3i(idx(0) >> 2, idx(1) >> 2, idx(2) >> 2)));
    if (bt == blocks_.end() || !(bt->second & bit(idx)))
      return nullptr;
    const auto it = voxels_.find(pack(idx));
    return it == voxels_.end() || it->second.n == 0 ? nullptr : &it->second;
  }

  // The test of one ray (origin to end) against one stored voxel (see carve).
  bool sees_through(Voxel &v, const V3D &origin, const V3D &end)
  {
    const double core2 = fs_.core * fs_.core, beyond2 = fs_.beyond * fs_.beyond;
    const V3D ray = end - origin;
    const double r = ray.norm();
    const V3D d = ray / r;
    Test &t = test(v);
    const Eigen::Vector3f e = t.Lc * (origin - t.mean.cast<double>()).cast<float>();
    const Eigen::Vector3f g = t.Lc * d.cast<float>();
    const float gg = g.squaredNorm(), eg = e.dot(g);
    const float t_closest = -eg / gg;
    if (t_closest < fs_.min_range || t_closest >= r || e.squaredNorm() - eg * eg / gg > core2)
      return false;   // the ray does not pass through it, or only behind its end
    if (on_surface(v, t))
    {
      // a surface: the end point must lie on its far side, beyond the noise across it (a ray ending on the same
      // surface farther on comes as close to the voxel's plane at grazing incidence)
      const Eigen::Vector3f &n = t.planar ? t.normal : t.pooled_normal;
      const float near = n.dot((origin - t.mean).cast<float>()), far = n.dot((end - t.mean).cast<float>());
      if (near * far >= 0 || far * far < beyond2 * (t.planar ? t.across2 : t.pooled_across2))
        return false;
    }
    else if (t.mahalanobis2(end) < beyond2)
      return false;   // it ends in it or just past it
    return true;
  }
  // Whether the voxel is (on) a surface: its own distribution is one, or else its mean lies on the surface of its
  // pooled neighbourhood (a voxel of a few scan lines of a floor seen at grazing incidence is a line, not a plane;
  // computed once per scan, the neighbours change).
  bool on_surface(const Voxel &v, Test &t)
  {
    if (t.planar)
      return true;
    if (t.pooled_scan != scan_)
    {
      t.pooled_scan = scan_;
      Distribution d;
      bool is_blob;
      V3D normal;
      double offset;
      t.pooled_surface = false;
      if (pooled_distribution(unpack(v.key), d, is_blob) && !is_blob && one_surface(d.cov, d.mean, normal, offset))
      {
        const double across = normal.dot(d.cov * normal), off = normal.dot(t.mean - d.mean);
        t.pooled_surface = off * off <= 4 * (across + fs_.ray_sigma * fs_.ray_sigma);   // the voxel lies on it
        t.pooled_normal = normal.cast<float>();
        t.pooled_across2 = across + fs_.ray_sigma * fs_.ray_sigma;
      }
    }
    return t.pooled_surface;
  }

  static uint64_t splitmix64(uint64_t x)
  {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  // Occupancy of 4 x 4 x 4 blocks of voxels, one bit per voxel: the ray casting skips empty space without a lookup.
  static uint64_t bit(const Eigen::Vector3i &idx)
  {
    return uint64_t(1) << ((idx(0) & 3) | (idx(1) & 3) << 2 | (idx(2) & 3) << 4);
  }
  void mark(const Eigen::Vector3i &idx, bool occupied)
  {
    const uint64_t key = pack(Eigen::Vector3i(idx(0) >> 2, idx(1) >> 2, idx(2) >> 2));
    if (occupied)
      blocks_[key] |= bit(idx);
    else
    {
      const auto it = blocks_.find(key);
      if (it != blocks_.end() && (it->second &= ~bit(idx)) == 0)
        blocks_.erase(it);
    }
  }
  void rebuild_blocks()
  {
    blocks_.clear();
    for (const auto &kv : voxels_)
      mark(unpack(kv.first), true);
  }

  // Calls f(voxel) for every stored voxel the ray origin + t d, t in [t0, t1], passes (3-D DDA over the grid). It
  // only reads the map: rays are walked in parallel.
  template <typename F>
  void traverse(const V3D &origin, const V3D &d, double t0, double t1, F f)
  {
    const V3D a = origin + t0 * d;
    Eigen::Vector3i idx = index_of(a);
    const Eigen::Vector3i last = index_of(origin + t1 * d);
    int step[3];
    double t_max[3], t_delta[3];
    for (int ax = 0; ax < 3; ax++)
    {
      if (d(ax) > 0)
        step[ax] = 1, t_max[ax] = ((idx(ax) + 1) * p_.size - a(ax)) / d(ax), t_delta[ax] = p_.size / d(ax);
      else if (d(ax) < 0)
        step[ax] = -1, t_max[ax] = (idx(ax) * p_.size - a(ax)) / d(ax), t_delta[ax] = -p_.size / d(ax);
      else
        step[ax] = 0, t_max[ax] = t_delta[ax] = std::numeric_limits<double>::infinity();
    }
    const double span = t1 - t0;
    uint64_t block_key = ~uint64_t(0), mask = 0;
    while (true)
    {
      const Eigen::Vector3i b(idx(0) >> 2, idx(1) >> 2, idx(2) >> 2);   // floor(idx / 4)
      if (pack(b) != block_key)
      {
        block_key = pack(b);
        const auto bt = blocks_.find(block_key);
        mask = bt == blocks_.end() ? 0 : bt->second;
      }
      if (mask & bit(idx))
      {
        const auto it = voxels_.find(pack(idx));
        if (it != voxels_.end())
          f(it->second);
      }
      if (idx == last)
        return;
      const int ax = t_max[0] < t_max[1] ? (t_max[0] < t_max[2] ? 0 : 2) : (t_max[1] < t_max[2] ? 1 : 2);
      if (t_max[ax] > span)
        return;
      idx(ax) += step[ax];
      t_max[ax] += t_delta[ax];
    }
  }

  // A voxel takes no more points once its position is known to freeze_sigma (0: never): the variance of its mean,
  // σ² Σw² / (Σw)², is at most freeze_sigma².
  bool frozen(const Voxel &v) const
  {
    if (p_.freeze_sigma <= 0 || v.w <= 0)
      return false;
    return v.w * v.w * p_.freeze_sigma * p_.freeze_sigma >= noise2() * v.w2;
  }

  // Updates a voxel's resolved flag from its statistics.
  void refresh(Voxel &v) const
  {
    if (v.n == 0)
    {
      v.resolved = false;
      return;
    }
    Distribution d;
    distribution(v.origin, v.w, v.w2, v.sum, v.sum_sq, d);
    v.resolved = resolved(d, p_.min_n_eff);
  }

  static constexpr const char *MAGIC = "VXMP";
  static constexpr uint32_t VERSION = 3;
  static constexpr uint32_t BYTE_ORDER_MARK = 0x01020304;
  static constexpr uint64_t VOXEL_BYTES = 8 + 4 + 8 + 8 + 3 * 8 + 9 * 8;   // a voxel's record in the file
  template <typename T>
  static void put(std::ofstream &f, const T &v) { f.write(reinterpret_cast<const char *>(&v), sizeof(T)); }
  template <typename T>
  static bool get(std::ifstream &f, T &v) { return bool(f.read(reinterpret_cast<char *>(&v), sizeof(T))); }

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

  Params p_;
  FreeSpace fs_;
  std::unordered_map<uint64_t, Voxel> voxels_;
  std::vector<Voxel *> touched_;
  std::vector<Voxel *> pending_;    // voxels seen through or confirmed in this scan (carve, insert, settle)
  struct Walk                       // one thread's share of a scan's rays (carve)
  {
    std::vector<std::pair<Voxel *, int>> passed;   // stored voxels on the rays, with the ray's index
    FreeCells::Misses cells;
    std::vector<Voxel *> cell_voxels;              // the stored voxel of each block of cells (band_k), or null
  };
  std::vector<Walk> walks_;
  std::vector<V3D> ray_ends_;
  std::vector<double> ray_margin_;
  std::vector<Voxel *> ray_voxel_;   // the stored voxel each ray ends in (graze_k), or null
  std::vector<Voxel *> prepared_, prepared_list_;   // prepare()'s voxels, and its input
  FreeCells cells_;
  std::vector<uint64_t> removed_;
  std::vector<Test> tests_;
  std::vector<int32_t> free_tests_;
  std::unordered_map<uint64_t, uint64_t> blocks_;   // block key -> occupied voxels (bit)
  uint32_t scan_ = 1;
};
