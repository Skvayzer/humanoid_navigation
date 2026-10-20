#pragma once
// Offline evaluation: feeds a .g1replay dataset (scripts/make_replay.py) straight into FAST_LIO's
// callbacks and processing step, without DDS or the 100 Hz timer, and writes the odometry as TUM.
// Messages, their order and the output files match robot/tools/replay_lockstep.py, so both give the
// same trajectory; this one only skips the transport and the waiting.
#include <zlib.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace replay
{
struct Hooks
{
  std::function<void(sensor_msgs::msg::Imu::UniquePtr)> imu;
  std::function<void(sensor_msgs::msg::PointCloud2::UniquePtr)> lidar;
  std::function<bool()> process;                           // processes one scan if possible
  std::function<bool(nav_msgs::msg::Odometry &)> new_pose;  // the pose of the scan just processed, if any
  std::function<bool()> ok;                                // false: stop early (Ctrl-C)
};

namespace detail
{
template <typename T>
bool read_pod(std::ifstream &f, T &v)
{
  return bool(f.read(reinterpret_cast<char *>(&v), sizeof(T)));
}

// Value of a top-level string or number key in the (flat, json.dumps) meta.
inline std::string meta_value(const std::string &json, const std::string &key)
{
  const auto k = json.find("\"" + key + "\"");
  if (k == std::string::npos)
    return "";
  auto p = json.find(':', k) + 1;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\n'))
    p++;
  if (json[p] == '"')
    return json.substr(p + 1, json.find('"', p + 1) - p - 1);
  return json.substr(p, json.find_first_of(",}\n", p) - p);
}

inline builtin_interfaces::msg::Time stamp(int64_t ns)
{
  builtin_interfaces::msg::Time t;
  t.sec = int32_t(ns / 1000000000);
  t.nanosec = uint32_t(ns % 1000000000);
  return t;
}

inline sensor_msgs::msg::PointField field(const char *name, uint32_t offset, uint8_t type)
{
  sensor_msgs::msg::PointField f;
  f.name = name;
  f.offset = offset;
  f.datatype = type;
  f.count = 1;
  return f;
}
}  // namespace detail

// One cloud and the IMU samples before it, in file order.
struct Batch
{
  std::vector<sensor_msgs::msg::Imu::UniquePtr> imu;
  sensor_msgs::msg::PointCloud2::UniquePtr cloud;  // null at the end of the file
};

// Returns 0 on success, 130 if stopped early (the poses so far are still written). Prints one JSON summary line (as replay_lockstep.py) and writes
// <out_tum without extension>_replay.json next to the TUM file.
inline int run(const std::string &path, const std::string &out_tum, const Hooks &hooks)
{
  using namespace detail;
  std::ifstream f(path, std::ios::binary);
  char magic[4];
  uint32_t version = 0, meta_len = 0;
  if (!f.read(magic, 4) || std::string(magic, 4) != "G1RP" || !read_pod(f, version) || version != 1 || !read_pod(f, meta_len))
  {
    fprintf(stderr, "%s: not a version-1 .g1replay file\n", path.c_str());
    return 2;
  }
  std::string meta(meta_len, '\0');
  f.read(&meta[0], meta_len);
  const std::string name = meta_value(meta, "name");
  const std::string frame = meta_value(meta, "frame_id").empty() ? "livox_frame" : meta_value(meta, "frame_id");
  const double duration = atof(meta_value(meta, "duration_s").c_str());
  const std::vector<sensor_msgs::msg::PointField> fields = {
      field("x", 0, 7), field("y", 4, 7), field("z", 8, 7), field("intensity", 12, 7), field("ring", 16, 4), field("time", 18, 7)};
  constexpr uint32_t POINT_STEP = 22;

  // A reader thread reads and decompresses ahead (zlib takes about as long as FAST_LIO's own processing),
  // handing over one batch per cloud through a short queue; this thread calls the hooks in file order.
  std::deque<Batch> queue;
  std::mutex mutex;
  std::condition_variable cv;
  bool finished = false, stop = false;
  std::string error;
  constexpr size_t MAX_QUEUED = 16;
  std::thread reader([&]() {
    Batch batch;
    std::vector<uint8_t> compressed;
    uint8_t type;
    int64_t stamp_ns;
    const auto hand_over = [&]() {  // false once the consumer has stopped
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&]() { return queue.size() < MAX_QUEUED || stop; });
      if (stop)
        return false;
      queue.push_back(std::move(batch));
      batch = Batch();
      cv.notify_all();
      return true;
    };
    while (read_pod(f, type) && read_pod(f, stamp_ns))
    {
      if (type == 1)
      {
        double v[6];
        if (!f.read(reinterpret_cast<char *>(v), sizeof(v)))
          break;
        auto msg = std::make_unique<sensor_msgs::msg::Imu>();
        msg->header.stamp = stamp(stamp_ns);
        msg->header.frame_id = frame;
        msg->linear_acceleration.x = v[0];
        msg->linear_acceleration.y = v[1];
        msg->linear_acceleration.z = v[2];
        msg->angular_velocity.x = v[3];
        msg->angular_velocity.y = v[4];
        msg->angular_velocity.z = v[5];
        batch.imu.push_back(std::move(msg));
        continue;
      }
      int64_t end_ns;
      uint32_t n, clen;
      if (!read_pod(f, end_ns) || !read_pod(f, n) || !read_pod(f, clen))
        break;
      compressed.resize(clen);
      if (!f.read(reinterpret_cast<char *>(compressed.data()), clen))
        break;
      auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
      msg->header.stamp = stamp(stamp_ns);
      msg->header.frame_id = frame;
      msg->height = 1;
      msg->width = n;
      msg->fields = fields;
      msg->is_bigendian = false;
      msg->point_step = POINT_STEP;
      msg->row_step = POINT_STEP * n;
      msg->is_dense = true;
      msg->data.resize(size_t(POINT_STEP) * n);
      uLongf out_len = msg->data.size();
      if (uncompress(msg->data.data(), &out_len, compressed.data(), clen) != Z_OK || out_len != msg->data.size())
      {
        error = "corrupt cloud record";
        break;
      }
      batch.cloud = std::move(msg);
      if (!hand_over())
        break;
    }
    if (!stop)
      hand_over();  // trailing IMU samples
    std::lock_guard<std::mutex> lock(mutex);
    finished = true;
    cv.notify_all();
  });

  std::vector<nav_msgs::msg::Odometry> poses;
  const auto drain = [&]() {
    while (hooks.process())
    {
      nav_msgs::msg::Odometry odom;
      if (hooks.new_pose(odom))
        poses.push_back(odom);
    }
  };
  size_t scans = 0;
  bool interrupted = false;
  const auto t_start = std::chrono::steady_clock::now();
  while (true)
  {
    if (hooks.ok && !hooks.ok())
    {
      interrupted = true;
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
      cv.notify_all();
      break;
    }
    Batch batch;
    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&]() { return !queue.empty() || finished; });
      if (queue.empty())
        break;
      batch = std::move(queue.front());
      queue.pop_front();
      cv.notify_all();
    }
    for (auto &imu : batch.imu)
    {
      hooks.imu(std::move(imu));
      drain();  // process every scan as soon as IMU data past its end is in
    }
    if (batch.cloud)
    {
      hooks.lidar(std::move(batch.cloud));
      scans++;
      drain();
    }
  }
  reader.join();
  if (!error.empty())
  {
    fprintf(stderr, "%s: %s\n", path.c_str(), error.c_str());
    return 2;
  }
  if (interrupted)
    fprintf(stderr, "replay stopped early: writing the %zu poses so far\n", poses.size());
  else
    drain();
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

  FILE *tum = fopen(out_tum.c_str(), "w");
  if (tum == nullptr)
  {
    fprintf(stderr, "cannot write %s\n", out_tum.c_str());
    return 2;
  }
  fprintf(tum, "# timestamp x y z qx qy qz qw  (%s, FAST_LIO /rv/odometry)\n", name.c_str());
  for (const auto &m : poses)
  {
    const auto &p = m.pose.pose.position;
    const auto &q = m.pose.pose.orientation;
    fprintf(tum, "%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n", m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
            p.x, p.y, p.z, q.x, q.y, q.z, q.w);
  }
  fclose(tum);

  char summary[512];
  snprintf(summary, sizeof(summary),
           "{\"dataset\": \"%s\", \"mode\": \"offline%s\", \"scans\": %zu, \"poses\": %zu, \"scans_without_pose\": %zu, "
           "\"wall_s\": %.1f, \"realtime_factor\": %.2f}",
           name.c_str(), interrupted ? ", stopped early" : "", scans, poses.size(), scans - poses.size(), wall, wall > 0 ? duration / wall : 0.0);
  printf("%s\n", summary);
  fflush(stdout);
  const std::string json_path = out_tum.substr(0, out_tum.rfind('.')) + "_replay.json";
  if (FILE *js = fopen(json_path.c_str(), "w"))
  {
    fprintf(js, "%s\n", summary);
    fclose(js);
  }
  return interrupted ? 130 : 0;
}
}  // namespace replay
