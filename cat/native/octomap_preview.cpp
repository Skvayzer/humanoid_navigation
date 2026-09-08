// A bounded OctoMap adapter for visualization. No ROS / robot command interfaces.
#include <octomap/OcTree.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

namespace {
thread_local std::string last_error;
struct Map {
  octomap::OcTree tree;
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
  return 0;
}
size_t cat_size(void* ptr) { return ptr ? static_cast<Map*>(ptr)->tree.size() : 0; }

// hit[i]=0 still clears the observed ray (e.g. a ground return), but never
// makes its endpoint occupied. Truncated rays never produce an obstacle hit.
int cat_insert(void* ptr, const float* xyz, const uint8_t* hit, size_t count,
               const float* origin, double max_range) {
  try {
    if (!ptr || !xyz || !hit || !origin || !finite3(origin) ||
        count > 100000 || !(max_range > 0 && max_range <= 10)) return -1;
    auto& tree = static_cast<Map*>(ptr)->tree;
    octomap::KeySet free_keys, occupied_keys;
    const octomap::point3d sensor(origin[0], origin[1], origin[2]);
    octomap::KeyRay ray;
    for (size_t i = 0; i < count; ++i) {
      const float* p = xyz + 3*i;
      if (!finite3(p)) continue;
      octomap::point3d end(p[0], p[1], p[2]);
      const double distance = (end - sensor).norm();
      if (distance < 0.05) continue;
      const bool truncated = distance > max_range;
      if (truncated) end = sensor + (end - sensor) * (max_range / distance);
      if (!tree.computeRayKeys(sensor, end, ray)) continue;
      free_keys.insert(ray.begin(), ray.end());
      octomap::OcTreeKey key;
      if (!tree.coordToKeyChecked(end, key)) continue;
      if (hit[i] && !truncated) occupied_keys.insert(key);
      else free_keys.insert(key);
    }
    // Within one scan, an actual hit wins over an intersecting free ray.
    for (const auto& key : free_keys)
      if (!occupied_keys.count(key)) tree.updateNode(key, false, true);
    for (const auto& key : occupied_keys) tree.updateNode(key, true, true);
    tree.updateInnerOccupancy();
    return 0;
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
    auto& tree = static_cast<Map*>(ptr)->tree;
    const double r = tree.getResolution();
    size_t index = 0;
    for (int x=0; x<nx; ++x) for (int y=0; y<ny; ++y) for (int z=0; z<nz; ++z) {
      auto* node = tree.search(lower[0]+(x+.5)*r, lower[1]+(y+.5)*r,
                               lower[2]+(z+.5)*r);
      output[index++] = !node ? 0 : (tree.isNodeOccupied(node) ? 2 : 1);
    }
    return 0;
  } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
}
