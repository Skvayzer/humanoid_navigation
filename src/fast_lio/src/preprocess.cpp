#include "preprocess.h"

#include <cmath>
#include <cstring>
#include <rclcpp/logging.hpp>

using sensor_msgs::msg::PointField;

namespace
{
const PointField *find_field(const sensor_msgs::msg::PointCloud2 &msg, const string &name)
{
  for (const auto &f : msg.fields)
    if (f.name == name)
      return &f;
  return nullptr;
}

template <typename T>
T read_as(const uint8_t *p)
{
  T v;
  memcpy(&v, p, sizeof(T));
  return v;
}

// Value of a numeric PointField (little-endian).
double read_number(const uint8_t *p, uint8_t datatype)
{
  switch (datatype)
  {
    case PointField::INT8: return read_as<int8_t>(p);
    case PointField::UINT8: return read_as<uint8_t>(p);
    case PointField::INT16: return read_as<int16_t>(p);
    case PointField::UINT16: return read_as<uint16_t>(p);
    case PointField::INT32: return read_as<int32_t>(p);
    case PointField::UINT32: return read_as<uint32_t>(p);
    case PointField::FLOAT32: return read_as<float>(p);
    default: return read_as<double>(p);  // FLOAT64
  }
}

bool is_numeric(const PointField *f)
{
  return f != nullptr && f->datatype >= PointField::INT8 && f->datatype <= PointField::FLOAT64;
}

uint64_t splitmix64(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// 53 random bits for point i of the scan with this seed (a uniform number in [0, 1) times 2^53).
uint64_t hash53(uint64_t seed, uint64_t i)
{
  return splitmix64(seed ^ splitmix64(i)) >> 11;
}

// Seed of one scan: its stamp, varied by the random_seed parameter (0: the stamp alone).
uint64_t scan_seed(const builtin_interfaces::msg::Time &t, uint64_t random_seed)
{
  return (uint64_t(t.sec) * 1000000000ULL + t.nanosec) ^ (random_seed * 0x9e3779b97f4a7c15ULL);
}
}  // namespace

// Random selection of point_keep_ratio of the points: a hash of the point index seeded with the scan
// stamp, so there is no fixed pattern (e.g. every 4th point of a Mid-360 is always the same laser).
// hash * 2^-53 < ratio  <=>  hash < ceil(ratio * 2^53), both exact in double.
bool Preprocess::keep(uint64_t seed, size_t i) const
{
  return point_keep_ratio >= 1.0 || hash53(seed, i) < keep_threshold_;
}

void Preprocess::update_keep_threshold()
{
  keep_threshold_ = point_keep_ratio >= 1.0 ? 0 : uint64_t(std::ceil(point_keep_ratio * 9007199254740992.0));
}

void Preprocess::process(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg, PointCloudXYZI::Ptr& pcl_out)
{
  update_keep_threshold();
  livox_handler(msg, *pcl_out);
}

void Preprocess::process(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloudXYZI::Ptr& pcl_out)
{
  update_keep_threshold();
  pointcloud2_handler(msg, *pcl_out);
}

// Points are tested before they are built (blind, then the keep hash) and go straight into `out`.
// No duplicate-point check: in Mid-360 data the only repeated points are the (0,0,0) no-returns, which
// the blind check removes.
void Preprocess::livox_handler(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg, PointCloudXYZI &out)
{
  out.clear();
  const size_t plsize = min<size_t>(msg->point_num, msg->points.size());
  out.points.reserve(size_t(plsize * min(1.0, point_keep_ratio * 1.1 + 0.01)));

  const uint64_t seed = scan_seed(msg->header.stamp, random_seed);
  for (size_t i = 0; i < plsize; i++)
  {
    const auto &p = msg->points[i];
    // Lines below N_SCANS, first and second return only (tag bits 4-5)
    if (p.line >= N_SCANS || ((p.tag & 0x30) != 0x10 && (p.tag & 0x30) != 0x00))
      continue;
    if (!(p.x * p.x + p.y * p.y + p.z * p.z > (blind * blind)) || !keep(seed, i))
      continue;
    out.points.emplace_back();
    PointType &q = out.points.back();
    q.x = p.x;
    q.y = p.y;
    q.z = p.z;
    q.intensity = p.reflectivity;
    q.curvature = p.offset_time / float(1000000);  // ns after the header stamp -> ms
  }
  out.width = out.points.size();
  out.height = 1;
}

void Preprocess::pointcloud2_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloudXYZI &out)
{
  out.clear();
  const size_t plsize = size_t(msg->width) * msg->height;
  if (plsize == 0)
    return;

  const PointField *fx = find_field(*msg, "x"), *fy = find_field(*msg, "y"), *fz = find_field(*msg, "z");
  const PointField *fi = find_field(*msg, "intensity"), *ft = find_field(*msg, time_field);
  const auto is_float = [](const PointField *f) { return f != nullptr && f->datatype == PointField::FLOAT32; };
  if (!is_float(fx) || !is_float(fy) || !is_float(fz) || !is_numeric(ft) || msg->is_bigendian)
  {
    static bool reported = false;
    if (!reported)
    {
      string fields;
      for (const auto &f : msg->fields)
        fields += " " + f.name + "(type " + to_string(f.datatype) + ")";
      RCLCPP_ERROR(rclcpp::get_logger("preprocess"),
                   "Unreadable PointCloud2: need little-endian float32 x/y/z and a numeric time field '%s' "
                   "(preprocess.time_field). Fields:%s. Scans are dropped.", time_field.c_str(), fields.c_str());
      reported = true;
    }
    return;
  }

  float time_unit_scale;  // time field unit -> ms
  double units_per_s;
  switch (time_unit)
  {
    case SEC: time_unit_scale = 1.e3f; units_per_s = 1.0; break;
    case MS: time_unit_scale = 1.f; units_per_s = 1.e3; break;
    case US: time_unit_scale = 1.e-3f; units_per_s = 1.e6; break;
    case NS: time_unit_scale = 1.e-6f; units_per_s = 1.e9; break;
    default: time_unit_scale = 1.f; units_per_s = 1.e3; break;
  }
  // Absolute time: subtract the header stamp, expressed in the field's unit.
  const double t0 = time_absolute ? msg->header.stamp.sec * units_per_s + msg->header.stamp.nanosec * (units_per_s * 1.e-9)
                                  : 0.0;
  const bool intensity_float = is_float(fi), intensity_numeric = is_numeric(fi), time_float = is_float(ft);
  const uint64_t seed = scan_seed(msg->header.stamp, random_seed);

  // Points are tested before they are built (blind, then the keep hash) and go straight into `out`.
  out.points.reserve(size_t(plsize * min(1.0, point_keep_ratio * 1.1 + 0.01)));
  for (size_t row = 0, i = 0; row < msg->height; row++)
  {
    const uint8_t *p = msg->data.data() + row * msg->row_step;
    for (size_t col = 0; col < msg->width; col++, i++, p += msg->point_step)
    {
      const float x = read_as<float>(p + fx->offset), y = read_as<float>(p + fy->offset), z = read_as<float>(p + fz->offset);
      if (!(x * x + y * y + z * z > (blind * blind)) || !keep(seed, i))
        continue;
      out.points.emplace_back();
      PointType &q = out.points.back();
      q.x = x;
      q.y = y;
      q.z = z;
      q.normal_x = 0;
      q.normal_y = 0;
      q.normal_z = 0;
      q.intensity = intensity_float ? read_as<float>(p + fi->offset)
                    : intensity_numeric ? float(read_number(p + fi->offset, fi->datatype)) : 0.f;
      const double t = time_float ? read_as<float>(p + ft->offset) : read_number(p + ft->offset, ft->datatype);
      q.curvature = float(t - t0) * time_unit_scale;  // curvature unit: ms
    }
  }
  out.width = out.points.size();
  out.height = 1;
}
