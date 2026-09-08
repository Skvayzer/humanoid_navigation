# Perception-only validation notes

## 2026-09-08 — occupancy continuity and close-range input

Progress checkpoint: `5ee310e` (intentionally marked WIP; not deployed).

Problems reproduced on the G1:

- Brief scan/TF skew made the previous preview publish empty clouds.
- Whole-OctoMap reset every 10 seconds discarded accumulated observations.
- FAST-LIO registered clouds had already removed returns closer than 0.5 m.
- A naive Python CustomMsg subscription created thousands of point objects,
  causing its TF receive queue to fall behind. A worker thread alone did not
  fix that. The raw CDR1/NumPy reader removes that allocation bottleneck.

Implemented exclusively under `cat/`:

- Short gaps retain the last visualization with invalid/stale diagnostics;
  sustained data loss still clears output. No fabricated TF fallback.
- Per-cell observation age and rolling bounds replace periodic whole-map reset.
- Raw `/livox/lidar` input with `min_range: 0.0`, validated return tags and times,
  zero-return rejection, calibrated extrinsics, and 10 ms pose-time bins.
- Close returns are prioritized before sampling and shown in `nearfield_points`.
- Bounded processing worker and direct native occupancy export.

Checks completed before deployment:

- 35 CAT tests passed on robot ARM64, including actual ROS serialization versus
  the decoder, endian/alignment/truncation checks, global TF reception in an
  isolated ROS domain, cell expiry/compaction, near-range returns, and no-motion
  source checks.
- All 7 repository verification tests passed, including preservation of the
  original SLAM/navigation algorithms and configuration.
- Final passive 36-second test: 36 PREVIEW_OK updates, all 36 occupied clouds
  nonempty, with history continuing past both 10 and 30 seconds. Processing
  took approximately 0.48–0.59 seconds/update. Observed 168–243 near-field
  returns per update inside the volume (returns may include robot/crane).
- Probes subscribed to real scans and TF but replaced cloud/marker/status
  publication with local inspection. No motion commands, TF, goals or changes
  to existing processes were sent. The motion-arm token remained absent.

These timing measurements were taken on the host in a temporary test directory,
alongside the existing SLAM/localization/rosbridge/old CAT container. They are
not a measurement under the new container's CPU quota or Foxglove transport.
The operator must rebuild/restart only CAT, then verify `PREVIEW_OK`, near-field
geometry and stable visualization. Docker sudo access was not available to the
remote development session. The previous running image was left untouched.

Still unvalidated: moving-robot alignment, moving-obstacle clearing, body
masking, hardware blind areas, and policy usability. This remains visualization
only; `policy_ready=false` and `motion_enabled=false` in every state.
