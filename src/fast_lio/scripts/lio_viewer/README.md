# LIO map debugger

A visual debugger for the map and matching design of this FAST_LIO: it shows the ikd-Tree, the voxel-map
distributions and which of them are one surface, runs the IEKF update of any scan of a record from the
real prior or from a perturbed one, and shows per point where it attached (a voxel distribution, or a
5-NN plane from the ikd-Tree) and where it failed.

The update runs on the real C++ code: `fastlio_mapping --replay ... --debug-port` (`src/debug_session.hpp`)
stops right before the update of the chosen scan, serves the maps and the scan, runs the update as often
as asked with the state and covariance restored afterwards, and continues the record on request. The
page talks to it through `server.py`. Verified: the zero-noise update equals the full run's pose to
< 1 µm, and a normal replay is bit-identical with the session code compiled in.

## Run

```bash
source <ws>/install/setup.bash            # the workspace that built fast_lio
python3 scripts/lio_viewer/server.py --replay <record.g1replay> --scan 200 --trajectory <odom.tum> --open
```

- `--replay`: any `.g1replay` (`scripts/make_replay.py`); `--scan`: where to start (1-based, replay order).
- `--config`: FAST_LIO params file (default `config/dataset_mid360.yaml`).
- `-p key:=value`: FAST_LIO parameters, repeatable (`map_type:=ikdtree`, `filter_size_surf:=0.5`, ...).
- `--trajectory odom.tum`: a full run of the same record, drawn on the minimap so a scan can be picked by
  clicking the trajectory.
- `--port` (8765), `--binary` (default: `fastlio_mapping` of the sourced workspace), `--workdir`
  (`lio_viewer_session`: the session's log).

Stepping forward (▶, `n`) continues the replay in place; going back restarts it (a few seconds, longer for long
records). Stdlib only; the browser loads three.js from a CDN.

## What is shown

| Layer | Meaning |
|---|---|
| ikd-Tree points | the map points (one per `filter_size_map` voxel), coloured by height |
| their voxels | the `filter_size_map` cells that hold a map point, within a radius of the lidar |
| voxel distributions | 2σ ellipsoid of every voxel-map voxel (the spread of its points plus the uncertainty of their mean); colour = class |
| surface planes | the voxels whose distribution is one surface (`voxel_map.{pooled_thickness,pooled_planarity}`) |
| voxel classes | one surface / not resolved (matched through the pooled neighbourhood) / too thick / not planar (λ0/λ1); hover shows the numbers |
| down points | the downsampled scan at the shown iteration, coloured by status: matched to a voxel distribution (green), matched to a 5-NN plane (blue), no / far neighbours, 5-point plane fit failed, residual too large |
| residuals | a line from each point to its 5-NN plane (length = the point-to-plane residual) or to the mean of its distribution |
| 5-NN links | the 5 nearest map points used for the plane fit (selected point or all) |
| frames | IMU prior (grey), start = prior + noise (yellow), posterior (green), posterior without noise (white) |

"Pose noise on the prior" moves the start pose: x y z in the world frame, roll pitch yaw as the filter's own
body-frame rotation error. The update then runs from there; the table lists every iteration (● = map
search, otherwise the planes are reused), the matched counts, the mean residual and the step size, and
the text under the sliders gives the distance between the posterior with and without noise.

"Matching parameters" change the session: the voxel-map parameters (`point_sigma`, `freeze_sigma`,
`shape_ratio`, `min_n_eff`, the pooled-surface test) refit every voxel and reload the map, the scan settings
re-downsample the scan (`filter_size_surf`, `scan_patch.min_points`), and `max_iteration`; `map_type` is fixed
at start. They hold until the replay restarts.

"Hide everything above lidar + h" cuts ceilings so indoor scenes are visible from above. Click a down
point for its history over the iterations and the voxel that gave its plane. `window.lio` in the browser
console gives the state, the camera and `lio.look(dx, dy, dz)` for scripted views.

## Session protocol (for other tools)

`fastlio_mapping --replay <in> <out.tum> --debug-port <p> --ros-args ...` listens on 127.0.0.1:p; text
commands, JSON replies, binary payloads: `goto`, `map [radius]`, `scan`, `update dx dy dz rx ry rz`,
`set key value`, `voxel_flags`, `poses`, `continue`, `quit`. Layouts are documented in
`ws/src/FAST_LIO/src/debug_session.hpp`; `server.py`'s `Engine` class is a Python client
(`Engine(args, workdir).goto(k)`, `.update([...])`, `.map(r)`), usable without the browser.
