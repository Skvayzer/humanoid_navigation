# G1 whole-body panels (Foxglove extension)

Two panels for `real_robot/foxglove/wholebody_layout_v3.2.json` (and v3):

- **ESDF height**: sets the layout variable `esdf_z` on every slider move (the stock Variable Slider waits until the
  slider stops), mouse wheel and arrow keys step one 4 cm layer; the layout's `esdf_slice` user script
  (`../esdf_slice.ts`) cuts that layer from the executor's compressed ESDF cube.
- **G1 controls**: operator buttons stand, balance, go, stop, reset and a large DAMP (`/g1_wb/operator`), obstacles
  virtual / real / both (`/g1_wb/obstacle_mode`, the current one outlined), the scene drop-down with top-view
  previews (`/g1_wb/scene`; the list and the geometry come from the layout's panel config, written by
  `../make_layout_v3.py`), and the operator speed slider (`/g1_wb/speed`, only for policies trained with an operator
  speed; the executor refuses it otherwise). The top line shows the executor's mode, speed and stop latch from
  `/g1_wb/status`; an orange line appears when the executor reports the head (LiDAR) off its URDF mount.

The String type name follows the connection (`std_msgs/msg/String` with foxglove_bridge, `std_msgs/String` with
rosbridge): it is taken from `/g1_wb/status`.

Build: `./pack.sh` -> `g1-wholebody-panels.foxe` (plain JavaScript, no build step). Install: drag the `.foxe` onto
the Foxglove window (or Settings -> Extensions -> install local), then import the layout.
