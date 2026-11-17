// Localization mode (localization.map_file), included by laserMapping.cpp: works on its globals.
//
// The scans are matched against a primary map loaded from a .vmap (frame map, never changed) and a live layer
// (voxel_map, frame init): the live layer takes the points the primary map has no voxel for (unknown space,
// new objects), stays within live_radius of the robot and is cleared at each (re)localization. Until the first
// localization the filter runs as SLAM on the live layer alone.
//
// The filter estimates the IMU in init, the gravity-aligned frame of the start, as in SLAM. The first
// localization fixes map -> init; a later one moves the state instead (a jump: laserMapping.cpp collects jumps in
// init -> odom, so odom -> imu stays continuous).
//
// Relocalization (relocalization.h): the last submap_scans scans, in the level frame of the newest, placed on the
// primary map near a guess (localization.initial_pose, or a pose on /initialpose), or, without a configured guess,
// anywhere on the map: once, as soon as the first submap is complete (if it fails, a guess is asked for). Live it
// runs on its own thread while the filter goes on; the offline replay waits for it, so a replay is repeatable.
#include <deque>
#include <future>
#include <mutex>
#include <relocalization.h>

namespace loc
{
constexpr double SUBMAP_VOXEL = 0.1;   // the submap is made of the points' centroids in voxels of this size [m]
constexpr int TRIM_EVERY = 10;         // scans between two trims of the live layer to live_radius

struct Job
{
  Relocalizer::Result res;
  Rigid init_L;   // the submap's level frame in init
  bool global;    // the search over the whole map
};

struct Localization
{
  bool enabled = false;
  VoxelMap prior;                       // the primary map (frame map)
  std::unique_ptr<Relocalizer> reloc;
  Relocalizer::Params params;
  int submap_scans = 10;
  double live_radius = 30.0;
  bool synchronous = false;             // relocalize within the scan (offline replay)
  bool localized = false;
  Rigid map_init;                       // map -> init, from the first localization
  struct Scan
  {
    Rigid pose;                         // the IMU in init
    std::vector<V3D> pts;               // IMU frame
  };
  std::deque<Scan> scans;               // the last submap_scans scans (processing thread)
  std::mutex mtx;                       // guards the guess (set from the /initialpose callback)
  bool has_guess = false;
  Relocalizer::Pose guess;              // frame map
  bool global_pending = false;          // the search over the whole map, not run yet (no configured guess)
  std::future<Job> job;
  bool active() const { return enabled && localized; }
} state;

// The best voxel distribution for a measurement at p (init): the primary map's if one passes the gate, else
// the live layer's (as VoxelMap::distribution_match; mean and covariance in init). In SLAM mode the live layer's.
inline bool match_distribution(const V3D &p, const M3D &meas_cov, VoxelMap::Distribution &out, double &d2,
                               bool *pooled, bool *blob)
{
  if (state.active())
  {
    const M3D &R = state.map_init.R;
    if (state.prior.distribution_match(state.map_init * p, R * meas_cov * R.transpose(), out, d2, pooled, blob) &&
        d2 < D2D_GATE)
    {
      out.mean = R.transpose() * (out.mean - state.map_init.t);
      out.cov = R.transpose() * out.cov * R;
      return true;
    }
  }
  return voxel_map.distribution_match(p, meas_cov, out, d2, pooled, blob);
}


// Whether a point (init) goes into the live layer: always, unless the primary map has a voxel there.
inline bool to_live_layer(const V3D &p)
{
  return !state.active() || !state.prior.occupied(state.map_init * p);
}

inline void set_guess(double x, double y, double yaw)
{
  std::lock_guard<std::mutex> lock(state.mtx);
  state.guess.x = x, state.guess.y = y, state.guess.yaw = yaw;
  state.has_guess = true;
}

// After each updated scan: keeps it for the submap, and starts a relocalization if a guess waits (or the search
// over the whole map has not run) and enough scans are kept. Returns what started: 0 nothing, 1 a search near a
// guess, 2 the search over the whole map.
inline int after_scan(const state_ikfom &x, const PointCloudXYZI &undistorted)
{
  Localization::Scan s;
  s.pose = Rigid{x.rot.toRotationMatrix(), x.pos};
  const M3D R_IL = x.offset_R_L_I.toRotationMatrix();
  s.pts.reserve(undistorted.size());
  for (const PointType &p : undistorted.points)
    s.pts.push_back(R_IL * V3D(p.x, p.y, p.z) + x.offset_T_L_I);
  state.scans.push_back(std::move(s));
  while (int(state.scans.size()) > state.submap_scans)
    state.scans.pop_front();

  Relocalizer::Pose guess;
  bool global = false;
  {
    std::lock_guard<std::mutex> lock(state.mtx);
    if (state.job.valid() || int(state.scans.size()) < state.submap_scans || !(state.has_guess || state.global_pending))
      return 0;
    global = !state.has_guess;   // a guess given meanwhile goes first
    guess = state.guess;
    state.has_guess = false;
    state.global_pending = false;
  }
  // the submap in the level frame L of the newest scan, SUBMAP_VOXEL centroids
  const Rigid &last = state.scans.back().pose;
  const double yaw = std::atan2(last.R(1, 0), last.R(0, 0));
  const Rigid init_L{Eigen::AngleAxisd(yaw, V3D::UnitZ()).toRotationMatrix(), last.t};
  const Rigid L_init = init_L.inverse();
  std::unordered_map<uint64_t, std::pair<V3D, int>> grid;
  const auto u = [](double v) { return uint64_t(int64_t(std::floor(v / SUBMAP_VOXEL)) + (1 << 20)) & 0x1FFFFF; };
  for (const Localization::Scan &sc : state.scans)
    for (const V3D &p : sc.pts)
    {
      const V3D l = L_init * (sc.pose * p);
      auto &cell = grid[(u(l(0)) << 42) | (u(l(1)) << 21) | u(l(2))];
      cell.first += l;
      cell.second++;
    }
  std::vector<V3D> submap;
  submap.reserve(grid.size());
  for (const auto &kv : grid)
    submap.push_back(kv.second.first / kv.second.second);
  std::sort(submap.begin(), submap.end(), [](const V3D &a, const V3D &b) {
    return a(0) < b(0) || (a(0) == b(0) && (a(1) < b(1) || (a(1) == b(1) && a(2) < b(2))));
  });
  double height = std::numeric_limits<double>::quiet_NaN();   // unknown: the mapping sensor's height
  Relocalizer::height_above_floor(submap, height);
  state.job = std::async(state.synchronous ? std::launch::deferred : std::launch::async,
                         [submap = std::move(submap), height, guess, init_L, global]() {
                           Job j;
                           j.init_L = init_L;
                           j.global = global;
                           j.res = global ? state.reloc->localize_global(submap, height)
                                          : state.reloc->localize(submap, height, guess);
                           return j;
                         });
  return global ? 2 : 1;
}

// A finished relocalization, if any (the offline replay waits for it).
inline bool finished(Job &out)
{
  if (!state.job.valid())
    return false;
  if (!state.synchronous && state.job.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    return false;
  out = state.job.get();
  return true;
}

// Applies a successful relocalization. The first fixes map -> init; a later one returns the correction of the
// state (in init: the old pose of the state to its new one), the identity otherwise. Clears the live layer.
inline Rigid apply(const Job &j)
{
  const Rigid map_L{Eigen::AngleAxisd(j.res.pose.yaw, V3D::UnitZ()).toRotationMatrix(),
                    V3D(j.res.pose.x, j.res.pose.y, j.res.pose.z)};
  const Rigid map_init = map_L * j.init_L.inverse();
  voxel_map.clear();
  if (!state.localized)
  {
    state.map_init = map_init;
    state.localized = true;
    return Rigid();
  }
  return state.map_init.inverse() * map_init;
}
}  // namespace loc
