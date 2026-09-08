// A bounded OctoMap adapter for visualization. No ROS / robot command interfaces.
#include <octomap/OcTree.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
thread_local std::string last_error;
struct Map {
  octomap::OcTree tree;
  std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash> observed;
  size_t retired = 0;
  bool bounded = false;
  octomap::OcTreeKey lower_key, upper_key;
  bool contains(const octomap::OcTreeKey& key) const {
    if (!bounded) return true;
    for (int i=0; i<3; ++i)
      if (key[i] < lower_key[i] || key[i] > upper_key[i]) return false;
    return true;
  }
  explicit Map(double resolution) : tree(resolution) {
    tree.setProbHit(0.7); tree.setProbMiss(0.4);
    tree.setClampingThresMin(0.12); tree.setClampingThresMax(0.97);
  }
};
bool finite3(const float* p) {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}
}

extern "C" {
const char* cat_error() { return last_error.c_str(); }
void* cat_create(double resolution) {
  try {
    if (!(resolution >= 0.01 && resolution <= 0.2)) return nullptr;
    return new Map(resolution);
  } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
void cat_destroy(void* ptr) { delete static_cast<Map*>(ptr); }
int cat_reset(void* ptr) {
  if (!ptr) return -1;
  static_cast<Map*>(ptr)->tree.clear();
  static_cast<Map*>(ptr)->observed.clear();
  static_cast<Map*>(ptr)->retired = 0;
  static_cast<Map*>(ptr)->bounded = false;
  return 0;
}
size_t cat_size(void* ptr) { return ptr ? static_cast<Map*>(ptr)->tree.size() : 0; }

// hit[i]=0 still clears the observed ray (e.g. a ground return), but never
// makes its endpoint occupied. Truncated rays never produce an obstacle hit.
int cat_insert(void* ptr, const float* xyz, const uint8_t* hit, size_t count,
               const float* origins, size_t origin_count, double max_range, double now) {
  try {
    if (!ptr || !xyz || !hit || !origins || !std::isfinite(now) ||
        (origin_count != 1 && origin_count != count) ||
        count > 100000 || !(max_range > 0 && max_range <= 10)) return -1;
    auto& map = *static_cast<Map*>(ptr);
    auto& tree = map.tree;
    octomap::KeySet free_keys, occupied_keys;
    octomap::KeyRay ray;
    for (size_t i = 0; i < count; ++i) {
      const float* p = xyz + 3*i;
      const float* origin = origins + (origin_count == 1 ? 0 : 3*i);
      if (!finite3(p) || !finite3(origin)) continue;
      const octomap::point3d sensor(origin[0], origin[1], origin[2]);
      octomap::point3d end(p[0], p[1], p[2]);
      const double distance = (end - sensor).norm();
      if (distance <= 1e-6) continue;  // invalid zero ray, not a blind-zone filter
      const bool truncated = distance > max_range;
      if (truncated) end = sensor + (end - sensor) * (max_range / distance);
      if (!tree.computeRayKeys(sensor, end, ray)) continue;
      for (const auto& key : ray) if (map.contains(key)) free_keys.insert(key);
      octomap::OcTreeKey key;
      if (!tree.coordToKeyChecked(end, key)) continue;
      if (!map.contains(key)) continue;
      if (hit[i] && !truncated) occupied_keys.insert(key);
      else free_keys.insert(key);
    }
    // Within one scan, an actual hit wins over an intersecting free ray.
    for (const auto& key : free_keys) {
      if (occupied_keys.count(key)) continue;
      if (!map.observed.count(key)) tree.setNodeValue(key, 0.0f, true);
      tree.updateNode(key, false, true);
      map.observed[key] = now;
    }
    for (const auto& key : occupied_keys) {
      if (!map.observed.count(key)) tree.setNodeValue(key, 0.0f, true);
      tree.updateNode(key, true, true);
      map.observed[key] = now;
    }
    tree.updateInnerOccupancy();
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// Expire cells individually to UNKNOWN, not free; keep re-observed cells.
// No whole-map periodic reset. OctoMap 1.9.3's individual deleteNode path can
// assert on empty child arrays. Retire keys logically and occasionally compact
// retained log-odds instead; do not modify/replace the host OctoMap library.
int cat_prune(void* ptr, const double* lower, const double* upper, double now, double ttl) {
  try {
    if (!ptr || !lower || !upper || !std::isfinite(now) || !std::isfinite(ttl) || !(ttl > 0)) return -1;
    for (int i=0; i<3; ++i)
      if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i] >= upper[i]) return -1;
    auto& map = *static_cast<Map*>(ptr);
    const double half = map.tree.getResolution()/2;
    if (!map.tree.coordToKeyChecked(lower[0]+half, lower[1]+half, lower[2]+half, map.lower_key) ||
        !map.tree.coordToKeyChecked(upper[0]-half, upper[1]-half, upper[2]-half, map.upper_key)) return -1;
    map.bounded = true;
    int removed = 0;
    for (auto it = map.observed.begin(); it != map.observed.end();) {
      const auto p = map.tree.keyToCoord(it->first);
      const bool outside = p.x() < lower[0] || p.x() >= upper[0] ||
                           p.y() < lower[1] || p.y() >= upper[1] ||
                           p.z() < lower[2] || p.z() >= upper[2];
      if (outside || now - it->second > ttl) {
        it = map.observed.erase(it);
        ++removed;
      } else { ++it; }
    }
    map.retired += removed;
    if (map.retired > std::max(size_t(1000), map.observed.size()/4) || map.tree.size() > 800000) {
      std::vector<std::pair<octomap::OcTreeKey, float>> retained;
      retained.reserve(map.observed.size());
      for (const auto& item : map.observed) {
        const auto* node = map.tree.search(item.first);
        if (node) retained.emplace_back(item.first, node->getLogOdds());
      }
      map.tree.clear();
      for (const auto& item : retained) map.tree.setNodeValue(item.first, item.second, true);
      map.tree.updateInnerOccupancy();
      map.retired = 0;
    }
    return removed;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

// C-order XYZ grid, cell centers = lower_corner + (index + .5)*resolution.
// 0=unknown, 1=observed free, 2=occupied. search() handles coarser leaves too.
int cat_export(void* ptr, const double* lower, int nx, int ny, int nz,
               uint8_t* output, size_t capacity) {
  try {
    if (!ptr || !lower || !output || nx < 1 || ny < 1 || nz < 1 ||
        nx > 256 || ny > 256 || nz > 128 ||
        capacity != size_t(nx)*size_t(ny)*size_t(nz)) return -1;
    for (int i=0; i<3; ++i) if (!std::isfinite(lower[i])) return -1;
    auto& map = *static_cast<Map*>(ptr);
    auto& tree = map.tree;
    const double r = tree.getResolution();
    size_t index = 0;
    for (int x=0; x<nx; ++x) for (int y=0; y<ny; ++y) for (int z=0; z<nz; ++z) {
      octomap::OcTreeKey key;
      const bool valid = tree.coordToKeyChecked(lower[0]+(x+.5)*r, lower[1]+(y+.5)*r,
                                                 lower[2]+(z+.5)*r, key);
      auto* node = valid && map.observed.count(key) ? tree.search(key) : nullptr;
      output[index++] = !node ? 0 : (tree.isNodeOccupied(node) ? 2 : 1);
    }
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
}
