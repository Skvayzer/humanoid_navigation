#!/usr/bin/env python3
"""Summarises and plots a record_restamp_check.py recording.

    plot_restamp_check.py check.csv [check.png] [title]

Each output sample is matched to its raw input by its (bit-identical) values; outputs are the raw
samples in order, minus repeats. Needs numpy and matplotlib (>= 3.1).
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

T = 0.005
RAW_COLOR = '#eb6834'      # categorical slot 2 (orange): raw vendor stamps
OUT_COLOR = '#2a78d6'      # categorical slot 1 (blue): restamped output
REP_COLOR = '#1baf7a'      # categorical slot 3 (aqua): output step across a dropped repeat
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'


def load(path):
    raw, out = [], []
    for line in open(path):
        p = line.strip().split(',')
        if len(p) != 9:
            continue
        row = (int(p[1]), int(p[2]), tuple(p[3:]))
        (raw if p[0] == 'R' else out).append(row)
    return raw, out


def match(raw, out, window=200):
    """Index of the raw sample each output came from (values are compared as text), or -1 if this
    recording missed that raw message."""
    idx, j = [], 0
    for _, _, v in out:
        k, end = j, min(len(raw), j + window)
        while k < end and raw[k][2] != v:
            k += 1
        if k < end:
            idx.append(k)
            j = k + 1
        else:
            idx.append(-1)
    return np.array(idx)


def step_kinds(idx, out_dt):
    """Why each output step is as long as it is: 'repeat' if the raw messages skipped in between
    were repeats (no new data), 'loss' if more slots passed than raw messages (data missing)."""
    raw_gap = np.diff(idx)
    slots = np.round(out_dt / (T * 1e3)).astype(int)
    kind = np.full(len(out_dt), 'normal', dtype=object)
    kind[raw_gap >= 2] = 'repeat'
    kind[slots > raw_gap] = 'loss'
    return kind


def plot_output_steps(ax, t, dt, kind, ms):
    """Output dt, with steps across dropped repeats and across lost samples marked separately."""
    n = kind == 'normal'
    ax.plot(t[n], dt[n], 'o', ms=ms, color=OUT_COLOR, label='restamped output')
    r = kind == 'repeat'
    if r.any():
        ax.plot(t[r], dt[r], 'o', ms=ms + 1, color=REP_COLOR,
                label='step across a dropped repeat (no new data)')
    lo = kind == 'loss'
    if lo.any():
        ax.plot(t[lo], dt[lo], 'x', ms=ms + 4, mew=1.5, color=INK,
                label='step across a missing sample')


def pct(x, ps=(0, 1, 50, 99, 100)):
    return '  '.join(f'p{p}={np.percentile(x, p):8.3f}' for p in ps)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    path = sys.argv[1]
    png = sys.argv[2] if len(sys.argv) > 2 else path.rsplit('.', 1)[0] + '.png'
    raw, out = load(path)
    idx = match(raw, out)
    unmatched = int(np.sum(idx < 0))
    out = [o for o, i in zip(out, idx) if i >= 0]
    idx = idx[idx >= 0]
    rs = np.array([r[0] for r in raw]) * 1e-9
    ra = np.array([r[1] for r in raw]) * 1e-9
    os_ = np.array([o[0] for o in out]) * 1e-9
    oa = np.array([o[1] for o in out]) * 1e-9
    t0 = rs[0]

    raw_dt = np.diff(rs) * 1e3
    out_dt = np.diff(os_) * 1e3
    jitter = (np.diff(os_) - np.round(np.diff(os_) / T) * T) * 1e3
    removed = (rs[idx] - os_) * 1e3
    added = (oa - ra[idx]) * 1e3
    repeats = sum(raw[i][2] == raw[i - 1][2] for i in range(1, len(raw)))
    span = rs[-1] - rs[0]
    kind = step_kinds(idx, out_dt)
    print(f'{path}: {len(raw)} raw, {len(out)} output matched ({unmatched} outputs whose raw copy '
          f'this recording missed), {repeats} repeats in raw, raw span {span:.2f} s -> '
          f'~{span / T + 1 - len(raw):.1f} raw messages missing here')
    print(f'  output steps: {np.sum(kind == "normal")} normal, {np.sum(kind == "repeat")} across '
          f'dropped repeats, {np.sum(kind == "loss")} across missing samples (lost upstream, or only '
          f'missing in this recording: compare with the node\'s "lost" counter)')
    print(f'  output strictly increasing: {bool(np.all(np.diff(os_) > 0))}')
    print(f'  raw dt [ms]          {pct(raw_dt)}')
    print(f'  output dt [ms]       {pct(out_dt)}')
    print(f'  output jitter [ms]   {pct(jitter)}   (dt minus whole periods)')
    print(f'  removed delay [ms]   {pct(removed)}   (raw stamp - output stamp)')
    print(f'  added latency [ms]   {pct(added)}   (arrival of output - arrival of raw)')

    title = sys.argv[3] if len(sys.argv) > 3 else 'IMU restamping on the G1'
    ms = 2.5 if len(out) < 20000 else 1.2
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(f'{title}: {span:.0f} s, {len(out)} output samples',
                 x=0.01, ha='left', fontsize=12, color=INK)

    ax = axes[0, 0]
    ax.plot(rs[1:] - t0, raw_dt, 'o', ms=ms, color=RAW_COLOR, alpha=0.7, label='raw vendor stamps')
    plot_output_steps(ax, os_[1:] - t0, out_dt, kind, ms)
    style(ax, f'Stamp spacing: raw {raw_dt.min():.1f}-{raw_dt.max():.1f} ms, output on the 5 ms grid',
          'dt [ms]')
    ax.set_xlabel('time [s]', color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)

    ax = axes[0, 1]
    ax.plot(os_ - t0, removed, 'o', ms=ms, color=OUT_COLOR)
    style(ax, 'Delay removed per sample (raw stamp - output stamp)', 'delay [ms]')
    ax.set_xlabel('time [s]', color=INK2, fontsize=9)

    # Zoom: 0.3 s around the largest raw step away from the start.
    k = int(np.argmax(np.where(rs[1:] - t0 > 2.0, raw_dt, 0))) + 1
    lo, hi = rs[k] - 0.15, rs[k] + 0.15
    ax = axes[1, 0]
    m_r = (rs[1:] >= lo) & (rs[1:] <= hi)
    m_o = (os_[1:] >= lo) & (os_[1:] <= hi)
    ax.plot(rs[1:][m_r] - t0, raw_dt[m_r], 'o-', ms=4, lw=1, color=RAW_COLOR,
            label='raw vendor stamps')
    ax.plot(os_[1:][m_o] - t0, out_dt[m_o], '-', lw=1, color=OUT_COLOR)
    plot_output_steps(ax, os_[1:][m_o] - t0, out_dt[m_o], kind[m_o], 4)
    style(ax, f'Zoom on the largest raw step ({raw_dt[k - 1]:.1f} ms)', 'dt [ms]')
    ax.set_xlabel('stamp [s]', color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)

    ax = axes[1, 1]
    bins = np.arange(0, max(2.0, np.percentile(added, 99.5)) + 0.05, 0.05)
    ax.hist(np.clip(added, 0, bins[-1]), bins=bins, color=OUT_COLOR, rwidth=0.85)
    style(ax, f'Delay added by the node (median {np.median(added):.2f} ms, '
              f'p99 {np.percentile(added, 99):.2f} ms)', 'samples')
    ax.set_xlabel('arrival of output - arrival of raw [ms]', color=INK2, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png, dpi=130, facecolor=SURFACE)
    print(f'  plot: {png}')


if __name__ == '__main__':
    main()
