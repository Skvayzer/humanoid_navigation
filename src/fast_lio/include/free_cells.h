#pragma once
// Free space for the voxel map (VoxelMap::FreeSpace::cell): evidence of occupancy (a count in units of one miss)
// in cells of the voxel grid, `subdivision` per voxel edge (4: 8.75 cm with 0.35 m voxels; 1: the voxel
// itself), also where the map has no voxel. Each scan, the cells its rays cross (short of
// their end points by `margin`) are missed once and the cells its points fall in are hit once; a hit outweighs a
// miss in the same scan (the occupied cells of a scan take precedence, as in OctoMap), so a cell that a thin pole or
// an edge shares with the free space next to it stays occupied. A cell missed by many scans and hit by none is free:
// points that fall there are what moved into free space (a person walking), kept out of the map until hits make the
// cell occupied again (a person who stays, a door that closed).
// A cell is free at 0 or below; a never-observed cell starts at free_after, a miss takes 1, a hit adds hit_weight
// (up to free_after_max, down to the depth from which reoccupy_after hits make it occupied again).
// Storage: per voxel position a block of subdivision^3 int8 counts in an open-addressing hash table.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <Eigen/Core>

class FreeCells
{
  public:
  struct Params
  {
    int subdivision = 4;        // cells per voxel edge, 1-4 (a block of subdivision^3 cells fits a 64-bit mask);
                                //   0: no cells
    int free_after = 7;         // misses (scans whose rays cross it, none of whose points fall in it) that make a
                                //   never-observed cell free: space crossed rarely (far, fast motion) turns free soon
    int free_after_max = 20;    // misses that make even a long-occupied cell free (the most evidence a cell holds)
    int reoccupy_after = 5;     // hits that make a long-free cell occupied again (a person who stops is let in
                                //   after 0.5 s)
    int hit_weight = 9;         // a hit outweighs this many misses (a weak miss keeps cells that a thin pole or an
                                //   edge shares with free space occupied); in the same scan a hit always wins
    double margin = 0.2;        // rays mark cells free up to this short of their end point [m]
  };

  void set(double voxel_size, const Params &p)
  {
    size_ = voxel_size;
    s_ = std::max(1, std::min(4, p.subdivision));
    cell_ = voxel_size / s_;
    p_ = p;
    hit_ = std::max(1, std::min(127, p.hit_weight));
    max_ = std::max(1, std::min(127, p.free_after_max));
    start_ = std::max(1, std::min(max_, p.free_after));
    min_ = std::max(-128, -hit_ * (std::max(1, p.reoccupy_after) - 1));
  }
  const Params &params() const { return p_; }
  size_t blocks() const { return blocks_.size(); }

  // Whether world point p lies in a free cell (as the previous scans left it).
  bool is_free(const Eigen::Vector3d &p) const
  {
    const Eigen::Vector3i c = cell_of(p);
    const int32_t b = find(pack(block_of(c)));
    return b >= 0 && blocks_[b].evidence[index_in_block(c)] <= 0;
  }

  // This scan's observations: a point at p (hit) ...
  void hit(const Eigen::Vector3d &p)
  {
    const Eigen::Vector3i c = cell_of(p);
    Block &b = touch(pack(block_of(c)));
    b.hits |= uint64_t(1) << index_in_block(c);
  }

  // The cells a ray crosses, per block: an existing block's index, or the key of one to create (index -1).
  struct Miss
  {
    int32_t index;
    uint64_t key, cells;
  };
  using Misses = std::vector<Miss>;

  // ... and a ray from origin along unit direction d over [t0, t1 - margin] (margin < 0: the parameter's): the cells
  // it crosses (misses), added to out. Reads only: rays are walked in parallel; add_misses() applies the lists.
  void walk(const Eigen::Vector3d &origin, const Eigen::Vector3d &d, double t0, double t1, Misses &out,
            double margin = -1) const
  {
    Miss m{-1, ~uint64_t(0), 0};
    cross(origin, d, t0, t1 - (margin < 0 ? p_.margin : margin), [&](uint64_t key, int cell) {
      if (key != m.key)
      {
        if (m.cells)
          out.push_back(m);
        m = Miss{find(key), key, 0};
      }
      m.cells |= uint64_t(1) << cell;
    });
    if (m.cells)
      out.push_back(m);
  }

  void add_misses(const Misses &misses)
  {
    for (const Miss &m : misses)
      (m.index >= 0 ? touch_index(m.index) : touch(m.key)).misses |= m.cells;
  }

  double cell_size() const { return cell_; }
  int subdivision() const { return s_; }   // cells per block (= voxel) edge; cell i of a block: i % s, i / s % s, i / s^2

  // Applies this scan's hits and misses (a hit outweighs a miss) and clears them; blocks in parallel.
  void update(int threads)
  {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int j = 0; j < int(touched_.size()); j++)
    {
      Block &b = blocks_[touched_[j]];
      for (uint64_t hit = b.hits; hit; hit &= hit - 1)
      {
        int8_t &e = b.evidence[__builtin_ctzll(hit)];
        e = int8_t(std::min(int(e) + hit_, max_));
      }
      for (uint64_t miss = b.misses & ~b.hits; miss; miss &= miss - 1)
      {
        int8_t &e = b.evidence[__builtin_ctzll(miss)];
        e = int8_t(std::max(int(e) - 1, min_));
      }
      b.hits = b.misses = 0;
      b.touched = false;
    }
    touched_.clear();
  }

  void clear()
  {
    blocks_.clear();
    keys_.assign(keys_.size(), EMPTY);
    touched_.clear();
  }

  // Drops the blocks farther than radius from centre (localization's live layer).
  void trim(const Eigen::Vector3d &centre, double radius)
  {
    std::vector<Block> kept;
    for (const Block &b : blocks_)
      if ((unpack(b.key).cast<double>() * size_ + Eigen::Vector3d::Constant(0.5 * size_) - centre).norm() <= radius)
        kept.push_back(b);
    blocks_.swap(kept);
    rehash(keys_.size());
  }

  private:
  struct Block
  {
    uint64_t key = 0;
    uint64_t hits = 0, misses = 0;   // this scan
    int8_t evidence[64];             // per cell, in misses (free at <= 0)
    bool touched = false;
  };
  static constexpr uint64_t EMPTY = ~uint64_t(0);   // pack() uses 63 bits

  // Calls f(block key, cell index) for every cell the ray crosses over [t0, t1] (3-D DDA over the cells).
  template <typename F>
  void cross(const Eigen::Vector3d &origin, const Eigen::Vector3d &d, double t0, double t1, F f) const
  {
    if (t1 <= t0)
      return;
    const Eigen::Vector3d a = origin + t0 * d;
    Eigen::Vector3i c = cell_of(a);
    const Eigen::Vector3i last = cell_of(origin + t1 * d);
    int step[3];
    double t_max[3], t_delta[3];
    for (int ax = 0; ax < 3; ax++)
    {
      if (d(ax) > 0)
        step[ax] = 1, t_max[ax] = ((c(ax) + 1) * cell_ - a(ax)) / d(ax), t_delta[ax] = cell_ / d(ax);
      else if (d(ax) < 0)
        step[ax] = -1, t_max[ax] = (c(ax) * cell_ - a(ax)) / d(ax), t_delta[ax] = -cell_ / d(ax);
      else
        step[ax] = 0, t_max[ax] = t_delta[ax] = 1e300;
    }
    const double span = t1 - t0;
    // the block and the cell's place in it follow the steps (no division per cell)
    Eigen::Vector3i b = block_of(c), r = c - b * s_;
    uint64_t key = pack(b);
    while (true)
    {
      f(key, r(0) + s_ * (r(1) + s_ * r(2)));
      if (c == last)
        return;
      const int ax = t_max[0] < t_max[1] ? (t_max[0] < t_max[2] ? 0 : 2) : (t_max[1] < t_max[2] ? 1 : 2);
      if (t_max[ax] > span)
        return;
      c(ax) += step[ax];
      t_max[ax] += t_delta[ax];
      r(ax) += step[ax];
      if (r(ax) == s_ || r(ax) < 0)
      {
        r(ax) -= step[ax] * s_;
        b(ax) += step[ax];
        key = pack(b);
      }
    }
  }

  Eigen::Vector3i cell_of(const Eigen::Vector3d &p) const
  {
    return Eigen::Vector3i(int(std::floor(p(0) / cell_)), int(std::floor(p(1) / cell_)), int(std::floor(p(2) / cell_)));
  }
  // floor(c / s) and c - s floor(c / s) per axis: the block (= voxel) of a cell and the cell's index in it
  int div(int v) const { return v >= 0 ? v / s_ : -((-v + s_ - 1) / s_); }
  Eigen::Vector3i block_of(const Eigen::Vector3i &c) const { return Eigen::Vector3i(div(c(0)), div(c(1)), div(c(2))); }
  int index_in_block(const Eigen::Vector3i &c) const
  {
    const Eigen::Vector3i r = c - block_of(c) * s_;
    return r(0) + s_ * (r(1) + s_ * r(2));
  }
  // as VoxelMap::pack: 21 bits per axis
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
  static uint64_t mix(uint64_t x)
  {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  int32_t find(uint64_t key) const
  {
    if (keys_.empty())
      return -1;
    const size_t mask = keys_.size() - 1;
    for (size_t s = mix(key) & mask;; s = (s + 1) & mask)
    {
      if (keys_[s] == key)
        return slots_[s];
      if (keys_[s] == EMPTY)
        return -1;
    }
  }

  Block &touch_index(int32_t i)
  {
    Block &b = blocks_[i];
    if (!b.touched)
    {
      b.touched = true;
      touched_.push_back(i);
    }
    return b;
  }

  // The block of key, created (all cells unknown) if new, marked as touched in this scan.
  Block &touch(uint64_t key)
  {
    if (2 * (blocks_.size() + 1) > keys_.size())
      rehash(std::max<size_t>(1024, 2 * keys_.size()));
    const size_t mask = keys_.size() - 1;
    size_t s = mix(key) & mask;
    while (keys_[s] != key && keys_[s] != EMPTY)
      s = (s + 1) & mask;
    if (keys_[s] == EMPTY)
    {
      keys_[s] = key;
      slots_[s] = int32_t(blocks_.size());
      blocks_.emplace_back();
      blocks_.back().key = key;
      std::fill(std::begin(blocks_.back().evidence), std::end(blocks_.back().evidence), int8_t(start_));   // never observed
    }
    Block &b = blocks_[slots_[s]];
    if (!b.touched)
    {
      b.touched = true;
      touched_.push_back(slots_[s]);
    }
    return b;
  }

  void rehash(size_t capacity)
  {
    size_t n = 1024;
    while (n < capacity || n < 2 * blocks_.size())
      n *= 2;
    keys_.assign(n, EMPTY);
    slots_.assign(n, -1);
    for (size_t i = 0; i < blocks_.size(); i++)
    {
      size_t s = mix(blocks_[i].key) & (n - 1);
      while (keys_[s] != EMPTY)
        s = (s + 1) & (n - 1);
      keys_[s] = blocks_[i].key;
      slots_[s] = int32_t(i);
    }
  }

  double size_ = 0.35, cell_ = 0.0875;
  int s_ = 4;
  Params p_;
  int hit_ = 9, max_ = 20, start_ = 7, min_ = -36;
  std::vector<Block> blocks_;
  std::vector<uint64_t> keys_;
  std::vector<int32_t> slots_;
  std::vector<int32_t> touched_;
};
