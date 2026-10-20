# g1_sensor_restamp

Republishes the G1 vendor Mid-360 IMU with **corrected timestamps**, so that the SLAM can trust `header.stamp`.

| | Topic | Stamp |
|---|---|---|
| in | `/utlidar/imu_livox_mid360` | Host time taken by the vendor process. It stalls for up to about 14 ms at every LiDAR scan, then releases the queued samples in a burst |
| out | `/g1/imu` | Time of the sample on the sensor's own 200 Hz grid. Repeated samples are dropped |

The background measurements are in [G1_Robot_Environment.md §5.3.1](../../../G1_Robot_Environment.md).

## How it works

The sensor samples on its own regular clock, and the vendor stamps each sample after a delay that is never negative. The grid is kept on the **lower envelope** of the stamps by a one-sided running minimum (passive synchronization, Olson 2010):

```
pred = t_prev + T + creep      creep lets the grid rise back onto the envelope
t    = min(stamp, pred)        late stamp: keep the grid; earlier stamp: the grid was too late
```

This gives the right answer for any delay as soon as the sample arrives. The grid can drop by at most T/2 per sample, so a start inside a catch-up burst can't compress the output.

Only a **loss** needs more work, because then the following samples stay late by about n·T instead of catching up:
- **Holding:** samples later than T/2 are held until a sample is on time again (they were only delayed), or until they have stayed late for 25 ms (samples were lost).
- **Placing each gap:** a gap goes right before a held sample if every later held sample is still late after the gaps already placed. This also handles two losses in one window.
- **Repeats:** a repeated sample (all six axes identical) keeps its slot but isn't published.

**Outage guard:**
- **Warning:** if no message arrives for `max_gap` (0.1 s, wall clock), the node warns once and logs when data comes back.
- **Restart:** if the stamps jumped by more than `max_gap`, the grid restarts at the first new sample, and **output continues immediately**, so the output gap equals the real gap.
- **Settling:** during the next 100 samples (0.5 s, logged as settling) the grid is still finding the envelope. The first stamps can be late by up to the delay of the first returning sample: typically under 2 ms, up to about 14 ms if it arrives inside a burst. They converge within a few samples.
- **Short gaps:** gaps below `max_gap` are handled as losses, with no restart.

## Validation on the robot (2026-10-20, G1 standing idle)

The node ran on PC2 against the live vendor stream. `record_restamp_check.py` recorded the input and the output side by side, and `plot_restamp_check.py` plotted them. The data and plots are in `results/imu_restamp/` in the project.

| Run | Raw stamp spacing | Output | Node counters |
|---|---|---|---|
| 30 s, IMU only | 1.5–8.4 ms (alternates about 4.3 / 5.7 ms) | on the 5 ms grid, jitter −0.14 / +0.002 ms, strictly increasing | 5 held, max delay 4.3 ms, 0 lost |
| 30 s, + cloud subscriber | 0.08–10.5 ms (a stall and burst every 100 ms) | the same | 145 held, max delay 6.0 ms, 0 lost |
| **10 min, + cloud subscriber** | 0.06–17.3 ms | the same, jitter −0.17 / +0.003 ms. All 8 370 steps of 10 ms are dropped repeats | 119 979 in, 0 lost, 0 restarts, max delay 13.8 ms |

- **What causes the stalls:** they appear only while something subscribes to the cloud. The vendor process stalls the IMU while it delivers each scan, so with the SLAM running the restamper is needed.
- **Added delay:** measured from Python, the node adds a median of 0.84 ms (p99 1.8 ms). That includes the extra DDS hop.
- **Losses:** the Python recorder missed a few raw messages (about 7 in 10 min) that the C++ node did receive.

## Validation (offline, `restamp_eval`)

| Input | Result |
|---|---|
| Real 60 s trace (G1 idle, 2026-10-20) | Output dt is a whole number of periods, −0.04 ms (p1) / −0.18 ms (worst), caused by the creep. Repeats dropped. All 10 samples missing from the trace were found (the span implies 12 001 slots, but 11 991 messages were recorded). The trace was recorded in Python, so these were most likely recorder drops, not vendor losses |
| Same trace with 40 injected loss events (5 seeds) | Every injected loss detected. 4–11 samples one slot off, all of them the ambiguous case below. All others within 0.1 ms of the lossless run |
| Same trace with a 2 s outage | Output gap = input gap. Settling error 1.0 ms (data back on time) / 4.9 ms (data back inside a burst), below 0.1 ms from the 3rd sample on. After that it matches the uninterrupted run exactly |
| Synthetic, 10 min, +50 ppm | Error vs. truth −0.04 / +0.07 ms (p1/p99), after a constant 0.24 ms (the minimum delay). 0 samples on a wrong slot. 287 of 287 injected losses detected |
| Synthetic, 1.5 s outage | Output gap = input gap. Settling error 1.6 ms / 10.7 ms, below 0.1 ms after 15 / 37 samples. Synthetic delays are uniform, so the minimum is found more slowly than on real data |
| Synthetic, −300 ppm (faster sensor) | Works for any creep |

**Added latency:**

| Case | Added latency |
|---|---|
| On-time samples (97 %) | 0 ms |
| Samples held during a stall | about 2 ms median |
| Samples near a loss | about 25–30 ms |
| Mean over all samples | about 0.1 ms |

**Known limits:**
- **Loss inside a catch-up burst:** the stamps there are compressed, so the exact position of the lost sample is ambiguous. 1–2 samples land one slot (5 ms) off, and after that the grid is exact again.
- **`creep` must exceed how much slower than `period` the sensor runs.**
  - The default of 1 µs per sample covers up to +200 ppm; the measured sensor is +20 ppm.
  - A sensor that is too slow shows up as `lost` growing steadily while the input is a clean 200 Hz. In that case raise `creep` or set `period`.
  - A smaller creep lowers the jitter: 0.5 µs gives −0.13 ms worst.

## Build, deploy, run

From the Mac (same flags as `robot/`):

```bash
rsync -rlpcv --exclude build --exclude install --exclude log --exclude .git --exclude __pycache__ ws/ G1-robot:~/workspace/romanvoronov/ws/
```

On the robot:

```bash
source ~/workspace/romanvoronov/env.sh
cd ~/workspace/romanvoronov/ws && colcon build --packages-select g1_sensor_restamp
source install/setup.bash
ros2 launch g1_sensor_restamp restamp.launch.py      # logs counters every 10 s
```

In FAST_LIO: `common.imu_topic: /g1/imu`.

## Parameters ([config/restamp.yaml](config/restamp.yaml))

| Parameter | Default | Purpose |
|---|---|---|
| `imu_in` / `imu_out` | `/utlidar/imu_livox_mid360` / `/g1/imu` | Input and output topics |
| `reliable_input` | `true` | The vendor publishes RELIABLE. Set `false` to compare loss counts |
| `acc_scale` | `1.0` | Accelerometer scale. `1.0` keeps the Livox convention (g), matching the evaluation bags. `9.80665` gives m/s² (REP-145) |
| `drop_duplicates` | `true` | Drop repeated samples |
| `period` | `0.005` | Nominal sample period [s] |
| `creep` | `1e-6` | Grid rise per sample [s]; see the known limits above |
| `max_gap` | `0.1` | [s] Quiet input triggers a warning; a larger stamp jump restarts the grid |

## Tools

```bash
ros2 run g1_sensor_restamp record_restamp_check.py 600 check.csv --cloud                 # input + output side by side (robot, node running)
ros2 run g1_sensor_restamp plot_restamp_check.py check.csv check.png "title"              # summary + plot (needs numpy, matplotlib)
ros2 run g1_sensor_restamp record_stamp_trace.py 60 ~/workspace/romanvoronov/trace.csv   # raw IMU trace (robot)
ros2 run g1_sensor_restamp restamp_eval trace.csv --drop-imu 40 --outage 2               # evaluate (anywhere)
ros2 run g1_sensor_restamp restamp_eval --synthetic --minutes 10 --ppm 50 --drop-imu 200 [--creep 0.5]
ros2 run g1_sensor_restamp record_scan_info.py 60 > scan_info.csv                        # cloud timing vs. IMU stalls
```

- **`restamp_eval`** has no ROS dependency. It also builds standalone: `c++ -std=c++17 -O2 -I include src/restamp_eval.cpp -o restamp_eval`.
  - `--out restamped.csv` writes the restamped stream: input index and stamp of each output sample.
  - The project's `scripts/restamp_replay.py` uses that to write `data/replay/<name>.restamped.g1replay`. Those copies have their IMU restamped exactly like `/g1/imu` on the robot.
- **`record_scan_info.py`** records the per-scan stamp, arrival time and point-time range together with the IMU stamps. Use it to study how the vendor's cloud processing blocks the IMU path.

- **`plot_restamp_check.py`** marks each output step that is longer than 5 ms by its cause. A **dropped repeat** means no new data. A **missing sample** means either lost upstream or missed only by the recorder; compare with the node's `lost` counter.

**Status:** built on the robot (Foxy, Release) and run against the live IMU for 10 min with a cloud subscriber: 0 losses, 0 restarts, output on the 5 ms grid. Still not tested: a walking robot and a real outage.
