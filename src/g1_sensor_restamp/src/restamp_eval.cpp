// Offline evaluation of the IMU restamper (no ROS). Feeds a recorded stamp trace, or a synthetic
// stream with known true sample times, through ImuRestamper and reports how regular the output is,
// how much delay was removed and whether injected losses were found.
//
//   restamp_eval <trace.csv> [--drop-imu K] [--outage SEC] [--creep US] [--seed S] [--dump out.csv]
//                [--out restamped.csv]
//   restamp_eval --synthetic [--minutes M] [--ppm P] [--drop-imu K] [--outage SEC] [--seed S]
//
// --out writes the restamped stream, one "<input index>,<stamp_ns>" row per output sample (repeats are
// dropped), so that other tools can apply exactly the robot's restamping offline.
//
// Trace rows (scripts/record_stamp_trace.py): "I,<stamp_ns>,ax,ay,az,gx,gy,gz"; other rows are
// ignored. In a recorded trace the true slots are unknown, so injected losses and outages are checked
// against the run on the unmodified trace: every remaining sample must keep its slot time.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "g1_sensor_restamp/restamper.hpp"

using namespace g1_sensor_restamp;
using Values = std::array<double, 6>;

namespace
{

ImuParams g_params;   // --creep overrides the default

struct Stream
{
  std::vector<double> s;       // raw stamps [s] relative to epoch
  std::vector<Values> v;
  std::vector<double> truth;   // true sample times (synthetic only)
  int64_t epoch_ns = 0;        // raw stamp of sample 0 (traces only)
};

double percentile(std::vector<double> x, double p)
{
  if (x.empty()) {return NAN;}
  std::sort(x.begin(), x.end());
  const double pos = p / 100.0 * (x.size() - 1);
  const size_t i = static_cast<size_t>(pos);
  const double f = pos - i;
  return i + 1 < x.size() ? x[i] * (1 - f) + x[i + 1] * f : x[i];
}

void print_ms(const char * name, const std::vector<double> & x)
{
  std::printf("  %-34s p0=%8.3f p1=%8.3f p50=%8.3f p99=%8.3f p100=%8.3f ms  (n=%zu)\n", name,
    1e3 * percentile(x, 0), 1e3 * percentile(x, 1), 1e3 * percentile(x, 50),
    1e3 * percentile(x, 99), 1e3 * percentile(x, 100), x.size());
}

void print_stats(const Stats & st)
{
  std::printf("  in %llu, out %llu, held %llu, lost %llu (%llu events), dup %llu, restarts %llu, "
    "max delay %.2f ms\n",
    (unsigned long long)st.received, (unsigned long long)st.emitted, (unsigned long long)st.held,
    (unsigned long long)st.lost, (unsigned long long)st.loss_events,
    (unsigned long long)st.duplicates, (unsigned long long)st.restarts, 1e3 * st.max_delay);
}

// Output slot time per input index (NaN if not output). hold: added latency per output sample, i.e.
// raw stamp of the message whose arrival released it minus its own raw stamp (raw stamps stand in
// for arrival times).
std::vector<double> run_imu(const Stream & in, const std::vector<char> & keep, Stats & st,
  std::vector<double> * hold = nullptr)
{
  std::vector<double> out(in.s.size(), NAN);
  size_t cur = 0;
  ImuRestamper<size_t> r(g_params, [&](double t, const size_t & i) {
      out[i] = t;
      if (hold) {hold->push_back(in.s[cur] - in.s[i]);}
    });
  for (size_t i = 0; i < in.s.size(); ++i) {
    if (keep[i]) {
      cur = i;
      r.push(in.s[i], in.v[i], i);
    }
  }
  st = r.stats();
  return out;
}

void print_hold(const std::vector<double> & hold)
{
  std::vector<double> held;
  for (double h : hold) {
    if (h > 0.0) {held.push_back(h);}
  }
  std::printf("  added latency: %.1f %% of output samples released on arrival (0 ms)\n",
    100.0 * (hold.size() - held.size()) / std::max<size_t>(hold.size(), 1));
  print_ms("added latency of held samples", held);
  double sum = 0.0;
  for (double h : hold) {sum += h;}
  std::printf("  mean added latency over all output samples: %.3f ms\n",
    1e3 * sum / std::max<size_t>(hold.size(), 1));
}

// K loss events of 1-3 consecutive samples, at least `gap` apart, away from both ends. Half of them
// hit the first late sample after a stall or the burst right behind it (the ambiguous case).
std::vector<char> choose_drops(const std::vector<double> & s, size_t margin, int K, size_t gap,
  std::mt19937 & rng, size_t & dropped)
{
  std::vector<char> keep(s.size(), 1);
  if (s.size() < 3 * margin) {return keep;}
  const size_t lo = margin, hi = s.size() - margin;
  std::vector<size_t> stall;
  for (size_t i = lo; i < hi; ++i) {
    if (s[i] - s[i - 1] > 0.009) {stall.push_back(i);}
  }
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::vector<size_t> used;
  dropped = 0;
  for (int e = 0, tries = 0; e < K && tries < 100 * K; ++tries) {
    const size_t i = (!stall.empty() && e % 2 == 0) ?
      stall[static_cast<size_t>(U(rng) * stall.size())] + static_cast<size_t>(U(rng) * 3) :
      lo + static_cast<size_t>(U(rng) * (hi - lo));
    bool ok = true;
    for (size_t u : used) {ok = ok && (i > u ? i - u : u - i) >= gap;}
    if (!ok) {continue;}
    const double x = U(rng);
    const size_t len = x < 0.7 ? 1 : (x < 0.9 ? 2 : 3);
    for (size_t j = 0; j < len; ++j) {keep[i + j] = 0;}
    used.push_back(i);
    dropped += len;
    ++e;
  }
  return keep;
}

// Compares two runs index by index: a slot error is a difference of more than half a period.
void compare(const std::vector<double> & a, const std::vector<double> & b, double T,
  size_t from = 0)
{
  size_t n = 0, slot_err = 0;
  double worst = 0.0;
  for (size_t i = from; i < a.size(); ++i) {
    if (std::isnan(a[i]) || std::isnan(b[i])) {continue;}
    ++n;
    const double d = std::abs(a[i] - b[i]);
    if (d > 0.5 * T) {
      ++slot_err;
    } else {
      worst = std::max(worst, d);
    }
  }
  std::printf("  vs reference run: %zu samples compared, %zu on a wrong slot, "
    "max |diff| otherwise %.4f ms\n", n, slot_err, 1e3 * worst);
}

std::vector<double> diffs(const std::vector<double> & x)
{
  std::vector<double> d;
  double prev = NAN;
  for (double v : x) {
    if (std::isnan(v)) {continue;}
    if (!std::isnan(prev)) {d.push_back(v - prev);}
    prev = v;
  }
  return d;
}

// Output dt minus the nearest whole number of periods (losses and duplicates leave longer steps).
std::vector<double> grid_jitter(const std::vector<double> & out, double T)
{
  std::vector<double> j;
  for (double d : diffs(out)) {j.push_back(d - std::round(d / T) * T);}
  return j;
}

std::vector<double> removed_delay(const std::vector<double> & raw, const std::vector<double> & out)
{
  std::vector<double> d;
  for (size_t i = 0; i < raw.size(); ++i) {
    if (!std::isnan(out[i])) {d.push_back(raw[i] - out[i]);}
  }
  return d;
}

bool load_trace(const std::string & path, Stream & imu)
{
  std::ifstream f(path);
  if (!f) {return false;}
  std::vector<int64_t> ns;
  std::string line;
  while (std::getline(f, line)) {
    if (line.size() < 3 || line[0] != 'I') {continue;}
    std::stringstream ss(line.substr(2));
    std::string tok;
    std::getline(ss, tok, ',');
    ns.push_back(std::stoll(tok));
    Values v{};
    for (double & x : v) {
      std::getline(ss, tok, ',');
      x = std::stod(tok);
    }
    imu.v.push_back(v);
  }
  for (int64_t x : ns) {imu.s.push_back(1e-9 * static_cast<double>(x - ns.front()));}
  if (!ns.empty()) {imu.epoch_ns = ns.front();}
  return !ns.empty();
}

// Fixed-rate samples with the measured G1 delay pattern: 0.2-1.9 ms ordinary delay, and every
// 100 ms with probability 0.3 a 3-14 ms stall whose samples are released in one burst; 6.8 % repeats.
Stream make_synthetic(double minutes, double ppm, std::mt19937 & rng)
{
  Stream st;
  const double T = 0.005 * (1.0 + 1e-6 * ppm);
  const size_t n = static_cast<size_t>(minutes * 60.0 / 0.005);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::normal_distribution<double> N(0.0, 0.01);
  st.truth.resize(n);
  st.s.resize(n);
  st.v.resize(n);
  for (size_t i = 0; i < n; ++i) {
    st.truth[i] = 10.0 + i * T;
    st.s[i] = st.truth[i] + 0.0002 + 0.0017 * U(rng);
  }
  for (size_t b = 7; b < n; b += 20) {   // a possible stall every 20 slots
    if (U(rng) >= 0.3) {continue;}
    const double release = st.truth[b] + 0.003 + 0.011 * U(rng);
    for (size_t i = b, j = 0; i < n && st.truth[i] < release; ++i, ++j) {
      st.s[i] = std::max(st.s[i], release + 0.00003 * j);
    }
  }
  for (size_t i = 1; i < n; ++i) {st.s[i] = std::max(st.s[i], st.s[i - 1] + 1e-6);}
  for (size_t i = 0; i < n; ++i) {
    if (i > 0 && U(rng) < 0.068) {
      st.v[i] = st.v[i - 1];
    } else {
      for (double & x : st.v[i]) {x = N(rng);}
    }
  }
  return st;
}

// Removes samples [from, to) and checks the restart: the output gap must equal the input gap, the
// stamps right after it converge, and from then on they match the uninterrupted run.
void outage_test(const Stream & imu, size_t from, size_t to, const std::vector<double> & out0,
  const char * label)
{
  const double T = g_params.period;
  std::vector<char> keep(imu.s.size(), 1);
  for (size_t i = from; i < to; ++i) {keep[i] = 0;}
  Stats st;
  const auto out = run_imu(imu, keep, st);
  std::printf("\nIMU outage of %.3f s, %s\n", imu.s[to] - imu.s[from - 1], label);
  print_stats(st);
  size_t before = from, after = to;
  while (before > 0 && std::isnan(out[before - 1])) {--before;}
  while (after < out.size() && std::isnan(out[after])) {++after;}
  std::printf("  gap: input %.3f s (raw stamps), output %.3f s\n", imu.s[to] - imu.s[from - 1],
    out[after] - out[before - 1]);
  // Error against the uninterrupted run while settling, and the sample from which it stays small.
  const size_t settle_end = std::min(out.size(), to + static_cast<size_t>(g_params.settle));
  double worst = 0.0;
  size_t converged = to;
  for (size_t i = to; i < settle_end; ++i) {
    if (std::isnan(out[i]) || std::isnan(out0[i])) {continue;}
    const double d = std::abs(out[i] - out0[i]);
    worst = std::max(worst, d);
    if (d > 1e-4) {converged = i + 1;}
  }
  std::printf("  settling: max error %.3f ms, below 0.1 ms from sample %zu after the restart on\n",
    1e3 * worst, converged - to);
  compare(out0, out, T, settle_end);
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string trace, dump, out_path;
  bool synthetic = false;
  int drop_imu = 0;
  unsigned seed = 1;
  double minutes = 10.0, ppm = 50.0, outage = 0.0;
  for (int a = 1; a < argc; ++a) {
    const std::string arg = argv[a];
    auto next = [&]() {return a + 1 < argc ? std::string(argv[++a]) : std::string("0");};
    if (arg == "--synthetic") {
      synthetic = true;
    } else if (arg == "--drop-imu") {
      drop_imu = std::stoi(next());
    } else if (arg == "--seed") {
      seed = static_cast<unsigned>(std::stoul(next()));
    } else if (arg == "--minutes") {
      minutes = std::stod(next());
    } else if (arg == "--ppm") {
      ppm = std::stod(next());
    } else if (arg == "--out") {
      out_path = next();
    } else if (arg == "--dump") {
      dump = next();
    } else if (arg == "--creep") {
      g_params.creep = 1e-6 * std::stod(next());
    } else if (arg == "--outage") {
      outage = std::stod(next());
    } else {
      trace = arg;
    }
  }
  std::mt19937 rng(seed);
  Stream imu;
  if (synthetic) {
    imu = make_synthetic(minutes, ppm, rng);
    std::printf("synthetic: %.1f min, %+.0f ppm clock offset, %zu IMU samples\n", minutes, ppm,
      imu.s.size());
  } else if (trace.empty() || !load_trace(trace, imu)) {
    std::fprintf(stderr,
      "usage: restamp_eval <trace.csv> [--drop-imu K] [--outage SEC] [--creep US] [--seed S] "
      "[--dump F]\n       restamp_eval --synthetic [--minutes M] [--ppm P] [--drop-imu K] "
      "[--outage SEC] [--creep US] [--seed S]\n");
    return 1;
  } else {
    std::printf("trace %s: %zu IMU samples, %.1f s\n", trace.c_str(), imu.s.size(),
      imu.s.back() - imu.s.front());
  }

  std::printf("\nIMU\n");
  const double T = g_params.period;
  Stats st;
  std::vector<double> hold0;
  const auto out0 = run_imu(imu, std::vector<char>(imu.s.size(), 1), st, &hold0);
  print_stats(st);
  print_hold(hold0);
  print_ms("raw dt", diffs(imu.s));
  print_ms("output dt", diffs(out0));
  print_ms("output dt - whole periods", grid_jitter(out0, T));
  print_ms("removed delay (raw - output)", removed_delay(imu.s, out0));
  if (synthetic) {
    std::vector<double> err;
    for (size_t i = 0; i < out0.size(); ++i) {
      if (!std::isnan(out0[i])) {err.push_back(out0[i] - imu.truth[i]);}
    }
    const double off = percentile(err, 50);
    size_t slot_err = 0;
    for (double & e : err) {
      e -= off;
      slot_err += std::abs(e) > 0.5 * T;
    }
    std::printf("  vs truth: constant offset %.3f ms, %zu on a wrong slot\n", 1e3 * off, slot_err);
    print_ms("error vs truth - offset", err);
  }
  if (!out_path.empty()) {   // the restamped stream: input index and corrected stamp of each output sample
    std::ofstream f(out_path);
    for (size_t i = 0; i < out0.size(); ++i) {
      if (!std::isnan(out0[i])) {f << i << ',' << imu.epoch_ns + std::llround(out0[i] * 1e9) << '\n';}
    }
  }
  if (!dump.empty()) {
    std::ofstream f(dump);
    f.precision(12);
    for (size_t i = 0; i < imu.s.size(); ++i) {f << i << ',' << imu.s[i] << ',' << out0[i] << '\n';}
  }

  if (drop_imu > 0) {
    std::printf("\nIMU with %d injected loss events\n", drop_imu);
    size_t dropped = 0;
    const auto keep = choose_drops(imu.s, 600, drop_imu, 80, rng, dropped);
    Stats st1;
    std::vector<double> hold1;
    const auto out1 = run_imu(imu, keep, st1, &hold1);
    print_stats(st1);
    print_hold(hold1);
    std::printf("  injected %zu missing samples, detected %llu (incl. %llu already in the input)\n",
      dropped, (unsigned long long)st1.lost, (unsigned long long)st.lost);
    compare(out0, out1, T);
  }

  if (outage > 0.0) {
    const size_t from = imu.s.size() / 2;
    size_t to = from;
    while (to < imu.s.size() && imu.s[to] < imu.s[from] + outage) {++to;}
    outage_test(imu, from, to, out0, "data returns on time");
    // Worst case for settling: the first sample back is the delayed head of a catch-up burst.
    while (to + 1 < imu.s.size() && imu.s[to] - imu.s[to - 1] < 0.009) {++to;}
    outage_test(imu, from, to, out0, "data returns with a catch-up burst");
  }
  return 0;
}
