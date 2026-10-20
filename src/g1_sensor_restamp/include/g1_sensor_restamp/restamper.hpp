#pragma once

// Timestamp reconstruction for the G1 Mid-360 IMU stream.
//
// The vendor stamps every sample with host time after a delay that is never negative: usually
// 0.2-2 ms, up to ~14 ms when its process stalls while handling a LiDAR scan (about every 100 ms),
// after which the queued samples arrive in a burst (G1_Robot_Environment.md, section 5.3.1). The sensor itself samples on
// a regular grid, so a sample's true time is the previous one plus one period. The grid is kept on
// the lower envelope of the stamps by a one-sided running minimum (passive synchronization, Olson
// 2010):
//
//   pred = t_prev + T + creep     creep lets the grid rise back onto the envelope
//   t    = min(stamp, pred)       late stamp: keep the grid; earlier stamp: the grid was too late
//
// That is right for any delay as soon as the sample arrives. Only a loss needs more: the following
// samples then stay late by ~n*T instead of catching up. So samples later than T/2 are held until a
// sample is on time again (they were only delayed) or they have stayed late for loss_window (n
// samples were lost before them: their slots move n periods on).
//
// A stamp jump beyond max_gap (outage, clock step) restarts the grid at the next stamp. Output
// continues immediately; for the first `settle` samples the grid is still finding the envelope, so
// the stamps can be late by up to the delay of the first sample and converge within a few samples.
//
// No ROS dependencies: times are seconds relative to an arbitrary epoch.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace g1_sensor_restamp
{

struct ImuParams
{
  double period = 0.005;        // [s] nominal sample period
  double creep = 1e-6;          // [s per sample] grid rise; covers a sensor up to 200 ppm slower
  double loss_window = 0.025;   // [s] how long late samples must stay late to count as a loss
  double max_gap = 0.1;         // [s] a larger stamp jump (outage, clock step) restarts the grid
  int settle = 100;             // samples after a (re)start reported as settling
  bool drop_duplicates = true;  // a repeated sample carries no new measurement
};

struct Stats
{
  uint64_t received = 0;
  uint64_t emitted = 0;
  uint64_t held = 0;          // arrived late and were held until resolved
  uint64_t loss_events = 0;
  uint64_t lost = 0;          // missing samples detected
  uint64_t duplicates = 0;
  uint64_t restarts = 0;      // stamp gap beyond max_gap
  double max_delay = 0.0;     // largest (stamp - slot time) of an output sample [s]
};

template<class Payload>
class ImuRestamper
{
public:
  using Values = std::array<double, 6>;
  using Emit = std::function<void (double, const Payload &)>;

  ImuRestamper(const ImuParams & p, Emit emit)
  : p_(p), emit_(std::move(emit)) {}

  // s: raw stamp [s]; v: acc + gyro, compared to spot repeats. emit() is called in slot order.
  void push(double s, const Values & v, const Payload & payload)
  {
    ++st_.received;
    const bool dup = have_prev_ && v == prev_;
    prev_ = v;
    have_prev_ = true;

    const double T = p_.period;
    const double pred = t_last_ + T + p_.creep;
    if (!started_ || std::abs(s - pred) > p_.max_gap) {
      restart();
      t_last_ = s;
      release(Sample{s, s, dup, payload});
      return;
    }
    // An earlier stamp pulls the grid down, by at most T/2 per sample, so that a start inside a
    // catch-up burst cannot compress the output.
    const Sample q{s, std::max(std::min(s, pred), t_last_ + 0.5 * T), dup, payload};
    t_last_ = q.t;
    if (s - pred < 0.5 * T) {   // on time: whatever is held was only delayed
      flush();
      release(q);
      return;
    }
    held_.push_back(q);
    ++st_.held;
    if (s - held_.front().s >= p_.loss_window) {resolve_losses();}
  }

  const Stats & stats() const {return st_;}
  bool settling() const {return started_ && settle_left_ > 0;}

private:
  struct Sample
  {
    double s;   // raw stamp
    double t;   // slot time
    bool dup;
    Payload payload;
  };

  // Samples were lost right before held sample i exactly when every held sample from i on is still
  // late after the gaps found so far: the lateness that never comes back down is n*T. Samples that
  // have stayed late for loss_window are released; newer ones stay held.
  void resolve_losses()
  {
    const double T = p_.period;
    suffix_min_.resize(held_.size());
    double m = std::numeric_limits<double>::infinity();
    for (size_t j = held_.size(); j-- > 0; ) {
      m = std::min(m, held_[j].s - held_[j].t);
      suffix_min_[j] = m;
    }
    double shift = 0.0;
    size_t confirmed = 0;
    for (size_t i = 0; i < held_.size(); ++i) {
      const bool old_enough = held_.back().s - held_[i].s >= p_.loss_window;
      if (old_enough && suffix_min_[i] - shift >= 0.5 * T) {
        const int64_t n = std::llround((suffix_min_[i] - shift) / T);
        shift += n * T;
        st_.lost += static_cast<uint64_t>(n);
        ++st_.loss_events;
      }
      held_[i].t += shift;
      confirmed += old_enough;
    }
    t_last_ += shift;
    for (size_t i = 0; i < confirmed; ++i) {
      release(held_.front());
      held_.pop_front();
    }
  }

  void restart()
  {
    if (started_) {
      flush();
      ++st_.restarts;
    }
    started_ = true;
    settle_left_ = p_.settle;
    last_out_ = -std::numeric_limits<double>::infinity();   // the output may jump with the clock
  }

  void flush()
  {
    for (const Sample & h : held_) {release(h);}
    held_.clear();
  }

  void release(const Sample & q)
  {
    if (settle_left_ > 0) {--settle_left_;}
    if (q.dup) {
      ++st_.duplicates;
      if (p_.drop_duplicates) {return;}
    }
    st_.max_delay = std::max(st_.max_delay, q.s - q.t);
    last_out_ = std::max(q.t, last_out_ + 1e-6);   // strictly increasing output
    ++st_.emitted;
    emit_(last_out_, q.payload);
  }

  ImuParams p_;
  Emit emit_;
  Stats st_;
  std::deque<Sample> held_;
  std::vector<double> suffix_min_;
  bool started_ = false;
  int settle_left_ = 0;
  double t_last_ = 0.0;   // slot time of the last sample taken in (held or released)
  double last_out_ = -std::numeric_limits<double>::infinity();
  Values prev_{};
  bool have_prev_ = false;
};

}  // namespace g1_sensor_restamp
