#pragma once
#include <cmath>
#include <deque>
#include <vector>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define INIT_MIN_SAMPLES (10)   // the initialization averages more than this many IMU samples

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

typedef esekfom::esekf<state_ikfom, 12, input_ikfom> Filter;

/// *************IMU propagation and scan deskewing
//
// Every IMU sample is integrated when it arrives (add_imu): the propagation filter takes one step (the mean
// of two consecutive samples; the first step after a scan starts at the scan end), and the step is kept with
// the state and covariance after it. A scan whose IMU data is complete is processed on the caller's filter,
// set to the state at the scan end (begin_scan): the steps up to the scan end give the deskew poses, and a
// partial step from the last of them reaches the end. The caller then updates its filter; end_scan makes the
// result the new base and integrates the newer samples again from it.
// Thread-safe: live, the IMU samples come in on their own thread while a scan is processed on another.
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum ScanStatus { INITIALISING, NO_IMU, DESKEWED };

  ImuProcess();

  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  // mapping.imu_init_time: the gravity and gyro bias are averaged over at least this long [s] (the sensor
  // must stand still); 0: the IMU samples of one scan.
  void set_init_time(double seconds) { init_time_ = seconds; }
  // mapping.acc_range, acc_satu_cov: an accelerometer reading at the sensor's range (|a| >= 0.99 range, in
  // the IMU's units; the Mid-360: 4 g) is only a bound on the true value. The steps that use it get the
  // noise acc_satu_cov [(m/s^2)^2] on that axis instead of cov_acc (Point-LIO leaves such readings out).
  // range 0: off.
  void set_acc_saturation(double range, double cov) { acc_range_ = range; acc_satu_cov_ = cov; }
  // mapping.init_cov_bg, init_cov_ba, init_cov_grav: the initial variances of the gyroscope bias [(rad/s)^2], the
  // accelerometer bias [(m/s^2)^2] and the gravity direction [rad^2] (FAST-LIO: 1e-4, 1e-3, 1e-5).
  void set_init_cov(double bg, double ba, double grav) { init_cov_bg_ = bg; init_cov_ba_ = ba; init_cov_grav_ = grav; }
  // Samples kept since the last scan end; beyond that the oldest go (into the base once integrated).
  void set_max_samples(size_t n) { max_samples_ = n; }
  size_t samples() const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return steps_.size();
  }
  Eigen::Matrix<double, 12, 12> Q;    // process noise, kept in step with the four covariances

  // A new IMU sample, in time order. Before the initialisation is complete it is only stored. Returns false
  // if the oldest sample had to be dropped (no scan for a long time).
  bool add_imu(const sensor_msgs::msg::Imu::ConstSharedPtr &imu);
  // The IMU stamps went backwards: forget the stored samples, back to the state at the last scan end.
  void clear_samples();
  // The first scan is not processed: its IMU samples are dropped.
  void discard_until(double end_time);
  // A scan [beg, end] whose IMU data is complete; kf holds the state at the last scan end. INITIALISING: its
  // samples went into the gravity and gyro-bias average (kf: the initial state). NO_IMU: no sample since the
  // last scan end, nothing done. DESKEWED: kf holds the prior at the scan end, and pcl_out the scan
  // deskewed to it.
  ScanStatus begin_scan(const PointCloudXYZI &scan, double beg, double end, Filter &kf, PointCloudXYZI &pcl_out);
  // After begin_scan and the caller's update, if any: kf holds the state at the scan end, which becomes the
  // base; the newer samples are integrated again from it.
  void end_scan(const Filter &kf);
  // The state at the newest sample if it is integrated (the IMU-rate odometry): stamp [s], state, covariance
  // and unbiased angular rate.
  bool newest_state(double &t, state_ikfom &x, Filter::cov &P, V3D &angvel) const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (steps_.empty() || !steps_.back().done)
      return false;
    const Step &s = steps_.back();
    t = s.t;
    x = s.x;
    P = s.P;
    angvel = s.angvel;
    return true;
  }
  // The unbiased angular rate of the last scan's last integrated sample (the scan-rate odometry's twist).
  V3D scan_angvel() const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return angvel_last;
  }
  // The IMU orientation (IMU to world) at the end of the initialisation, once it is done: the gravity
  // alignment, so its rows are the world axes in IMU coordinates.
  M3D initial_rotation() const
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return init_rot_;
  }

  V3D cov_acc;        // IMU noise and bias random walk, from the config (mapping.*_cov)
  V3D cov_gyr;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;

 private:
  // One IMU sample after the base, and the filter step that ends at it.
  struct Step
  {
    sensor_msgs::msg::Imu::ConstSharedPtr imu;
    double t;                 // stamp [s]
    bool done = false;        // integrated (not while initialising, and not before the last scan end)
    input_ikfom in;
    state_ikfom x;            // after the step
    Filter::cov P;
    V3D acc_s, angvel;        // world-frame acceleration without gravity, unbiased angular rate
    uint8_t satu = 0;         // accelerometer axes at the range limit in the step's samples (bits 0-2)
  };

  void propagate(Step &s, const sensor_msgs::msg::Imu::ConstSharedPtr &head);
  size_t scan_samples(double end) const;   // the leading samples that belong to a scan ending at `end`
  void IMU_init(size_t n, Filter &kf_state, int &N);
  void update_Q();
  Eigen::Matrix<double, 12, 12> step_Q(uint8_t satu) const;

  mutable std::mutex mtx_;                       // guards everything below
  Filter kf_;                                    // the propagation filter: at the newest integrated step
  std::deque<Step> steps_;                       // samples after the base, oldest first
  size_t scan_steps_ = 0;                        // of them: the deskewed scan's (begin_scan to end_scan)
  double scan_end_ = 0.0;
  bool   in_scan_ = false;                       // between begin_scan and end_scan
  size_t max_samples_ = 200;
  double acc_range_ = 0.0, acc_satu_cov_ = 30.0;
  double init_cov_bg_ = 0.0001, init_cov_ba_ = 0.001, init_cov_grav_ = 0.00001;
  state_ikfom base_x_;                           // the state at the last scan end
  Filter::cov base_P_;
  sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;   // the last sample at or before the base
  vector<Pose6D> IMUpose;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;    // of the last integrated step at or before the base
  V3D acc_s_last;
  double last_lidar_end_time_ = 0.0;
  int    init_iter_num = 1;
  double init_time_ = 0.0, init_first_stamp_ = 0.0;
  bool   b_first_frame_ = true;
  bool   init_done_ = false;      // enough samples averaged (begin_scan) ...
  bool   imu_need_init_ = true;   // ... and the initial state is the base (end_scan): samples are integrated
  M3D    init_rot_ = Eye3d;
};

ImuProcess::ImuProcess()
{
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  update_Q();
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  acc_s_last      = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::msg::Imu());
  double no_limit[23] = {0};
  kf_.init_dyn_share(get_f, df_dx, df_dw, nullptr, 0, no_limit);   // predict only: no measurement model
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::update_Q()
{
  Q.setZero();
  Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
  Q.block<3, 3>(3, 3).diagonal() = cov_acc;
  Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
  Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
}

void ImuProcess::set_gyr_cov(const V3D &cov)
{
  cov_gyr = cov;
  update_Q();
}

void ImuProcess::set_acc_cov(const V3D &cov)
{
  cov_acc = cov;
  update_Q();
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
  update_Q();
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
  update_Q();
}

void ImuProcess::IMU_init(size_t n, Filter &kf_state, int &N)
{
  /** 1. initializing the gravity and gyro bias (mean of the samples)
   ** 2. normalize the acceleration measurenments to unit gravity **/

  V3D cur_acc, cur_gyr;

  if (b_first_frame_)
  {
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = steps_.front().imu->linear_acceleration;
    const auto &gyr_acc = steps_.front().imu->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    init_first_stamp_ = steps_.front().t;
  }

  for (size_t i = 0; i < n; i++)
  {
    const auto &imu_acc = steps_[i].imu->linear_acceleration;
    const auto &gyr_acc = steps_[i].imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    N ++;
  }
  state_ikfom init_state = kf_state.get_x();
  // Gravity alignment: the world frame is z-up from the start. Its z axis is the measured "up" (the mean
  // specific force), its x axis the sensor's x axis projected onto the horizontal plane (the y axis if
  // the sensor's x is nearly vertical), so gravity is (0, 0, -g).
  const V3D z_w = mean_acc.normalized();
  V3D x_w = V3D::UnitX() - z_w.x() * z_w;
  if (x_w.norm() < 0.1)
    x_w = V3D::UnitY() - z_w.y() * z_w;
  x_w.normalize();
  M3D R_wi;   // IMU to world: the rows are the world axes in IMU coordinates
  R_wi.row(0) = x_w.transpose();
  R_wi.row(1) = z_w.cross(x_w).transpose();
  R_wi.row(2) = z_w.transpose();
  init_state.rot = SO3(R_wi);
  init_state.grav = S2(V3D(0, 0, -G_m_s2));
  init_state.bg  = mean_gyr;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  Filter::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = init_cov_bg_;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = init_cov_ba_;
  init_P(21,21) = init_P(22,22) = init_cov_grav_;
  kf_state.change_P(init_P);
}

// The process noise of a step whose samples are at the accelerometer's range on the axes in `satu`.
Eigen::Matrix<double, 12, 12> ImuProcess::step_Q(uint8_t satu) const
{
  Eigen::Matrix<double, 12, 12> q = Q;
  for (int k = 0; k < 3; k++)
    if (satu & (1 << k))
      q(3 + k, 3 + k) = acc_satu_cov_;
  return q;
}

// One step of the propagation filter from `head` to s (the previous sample and this one).
void ImuProcess::propagate(Step &s, const sensor_msgs::msg::Imu::ConstSharedPtr &head)
{
  const double head_stamp = rclcpp::Time(head->header.stamp).seconds();
  s.done = false;
  if (s.t < last_lidar_end_time_)
    return;
  const auto &tail = s.imu;
  V3D angvel_avr, acc_avr;
  angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
              0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
              0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
  acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
              0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
              0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

  acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

  // the first step after a scan starts at the scan end
  double dt = head_stamp < last_lidar_end_time_ ? s.t - last_lidar_end_time_ : s.t - head_stamp;

  s.in.acc = acc_avr;
  s.in.gyro = angvel_avr;
  s.satu = 0;
  if (acc_range_ > 0)
  {
    const double lim = 0.99 * acc_range_;
    const double h[3] = {head->linear_acceleration.x, head->linear_acceleration.y, head->linear_acceleration.z};
    const double t[3] = {tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z};
    for (int k = 0; k < 3; k++)
      if (fabs(h[k]) >= lim || fabs(t[k]) >= lim)
        s.satu |= 1 << k;
  }
  auto q = step_Q(s.satu);
  kf_.predict(dt, q, s.in);

  s.x = kf_.get_x();
  s.P = kf_.get_P();
  s.angvel = angvel_avr - s.x.bg;
  s.acc_s  = s.x.rot * (acc_avr - s.x.ba);
  for(int i=0; i<3; i++)
  {
    s.acc_s[i] += s.x.grav[i];
  }
  s.done = true;
}

bool ImuProcess::add_imu(const sensor_msgs::msg::Imu::ConstSharedPtr &imu)
{
  Step s;
  s.imu = imu;
  s.t = rclcpp::Time(imu->header.stamp).seconds();
  std::lock_guard<std::mutex> lock(mtx_);
  steps_.push_back(s);
  if (!imu_need_init_)
    propagate(steps_.back(), steps_.size() > 1 ? steps_[steps_.size() - 2].imu : last_imu_);
  if (steps_.size() <= max_samples_ || in_scan_)
    return true;
  // No scan for a long time: the oldest sample goes (once integrated, it becomes the base).
  const Step &old = steps_.front();
  if (old.done)
  {
    base_x_ = old.x;
    base_P_ = old.P;
    acc_s_last = old.acc_s;
    angvel_last = old.angvel;
  }
  last_imu_ = old.imu;
  steps_.pop_front();
  return false;
}

void ImuProcess::clear_samples()
{
  std::lock_guard<std::mutex> lock(mtx_);
  steps_.clear();
  scan_steps_ = 0;   // a scan in progress loses its samples too
  if (!imu_need_init_)
  {
    kf_.change_x(base_x_);
    kf_.change_P(base_P_);
  }
}

// The samples of a scan: taken while the previous one was before the scan end, and up to the end itself.
size_t ImuProcess::scan_samples(double end) const
{
  size_t n = 0;
  double imu_time = steps_.empty() ? end : steps_.front().t;
  while (n < steps_.size() && imu_time < end)
  {
    imu_time = steps_[n].t;
    if (imu_time > end)
      break;
    n++;
  }
  return n;
}

void ImuProcess::discard_until(double end_time)
{
  std::lock_guard<std::mutex> lock(mtx_);
  steps_.erase(steps_.begin(), steps_.begin() + scan_samples(end_time));
}

ImuProcess::ScanStatus ImuProcess::begin_scan(const PointCloudXYZI &scan, double beg, double end, Filter &kf,
                                              PointCloudXYZI &pcl_out)
{
  std::unique_lock<std::mutex> lock(mtx_);
  in_scan_ = true;
  scan_steps_ = scan_samples(end);
  if (scan_steps_ == 0)
    return NO_IMU;
  const Step &scan_last = steps_[scan_steps_ - 1];

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(scan_steps_, kf, init_iter_num);
    last_imu_ = scan_last.imu;
    const double span = scan_last.t - init_first_stamp_;
    steps_.erase(steps_.begin(), steps_.begin() + scan_steps_);
    scan_steps_ = 0;
    if (init_iter_num > INIT_MIN_SAMPLES && span >= init_time_)
    {
      init_done_ = true;
      const M3D R = kf.get_x().rot.toRotationMatrix();   // IMU to the gravity-aligned world
      std::cout << "IMU Initial Done (" << init_iter_num - 1 << " samples, " << span << " s); IMU roll "
                << atan2(R(2, 1), R(2, 2)) * 180 / M_PI << ", pitch " << -asin(R(2, 0)) * 180 / M_PI << " deg"
                << std::endl;
    }
    return INITIALISING;
  }

  /*** the IMU poses of the scan: the base, then each integrated step up to the scan end ***/
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, base_x_.vel, base_x_.pos, base_x_.rot.toRotationMatrix()));
  input_ikfom in;
  uint8_t satu = 0;
  const Step *last_done = nullptr;
  for (size_t i = 0; i < scan_steps_; i++)
  {
    const Step &s = steps_[i];
    if (!s.done)
      continue;
    IMUpose.push_back(set_pose6d(s.t - beg, s.acc_s, s.angvel, s.x.vel, s.x.pos, s.x.rot.toRotationMatrix()));
    in = s.in;
    satu = s.satu;
    acc_s_last = s.acc_s;
    angvel_last = s.angvel;
    last_done = &s;
  }
  state_ikfom x = last_done ? last_done->x : base_x_;
  Filter::cov P = last_done ? last_done->P : base_P_;
  const double imu_end_time = scan_last.t;
  scan_end_ = end;
  lock.unlock();

  /*** from the last sample of the scan to the scan end ***/
  kf.change_x(x);
  kf.change_P(P);
  double note = end > imu_end_time ? 1.0 : -1.0;
  double dt = note * (end - imu_end_time);
  auto q = step_Q(satu);
  kf.predict(dt, q, in);

  state_ikfom imu_state = kf.get_x();

  /*** sort point clouds by offset time ***/
  pcl_out = scan;
  // Some drivers' scans are not quite in time order; a scan that is needs no sort.
  if (!std::is_sorted(pcl_out.points.begin(), pcl_out.points.end(), time_list))
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return DESKEWED;
  // P_compensate = R_LI' (R_end' (R_i (R_LI P + t_LI) + T_ei) - t_LI), with the per-scan parts as matrices
  const M3D R_LI = imu_state.offset_R_L_I.toRotationMatrix();
  const V3D t_LI = imu_state.offset_T_L_I;
  const M3D A = R_LI.transpose() * imu_state.rot.toRotationMatrix().transpose();
  const V3D c = R_LI.transpose() * t_LI;
  M3D R_imu;
  V3D acc_imu, vel_imu, pos_imu, angvel_avr;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));

      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      V3D P_compensate = A * (R_i * (R_LI * P_i + t_LI) + T_ei) - c;

      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
  return DESKEWED;
}

void ImuProcess::end_scan(const Filter &kf)
{
  std::lock_guard<std::mutex> lock(mtx_);
  in_scan_ = false;
  if (!init_done_)
    return;
  if (imu_need_init_)
    init_rot_ = kf.get_x().rot.toRotationMatrix();
  imu_need_init_ = false;
  if (scan_steps_ > 0)   // a deskewed scan: its samples go
  {
    last_imu_ = steps_[scan_steps_ - 1].imu;
    last_lidar_end_time_ = scan_end_;
    steps_.erase(steps_.begin(), steps_.begin() + scan_steps_);
    scan_steps_ = 0;
  }
  base_x_ = kf.get_x();
  base_P_ = kf.get_P();
  kf_.change_x(base_x_);
  kf_.change_P(base_P_);
  for (size_t i = 0; i < steps_.size(); i++)   // the newer samples, again from the scan's state
    propagate(steps_[i], i == 0 ? last_imu_ : steps_[i - 1].imu);
}
