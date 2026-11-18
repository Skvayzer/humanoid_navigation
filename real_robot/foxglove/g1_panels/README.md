# G1 whole-body panels (Foxglove extension)

Two panels for `real_robot/foxglove/wholebody_layout_v2.json`:

- **ESDF height**: sets the layout variable `esdf_z` on every slider move (the stock Variable Slider waits until the
  slider stops), mouse wheel and arrow keys step one 4 cm layer; the `esdf_slice` user script cuts that layer.
- **G1 controls**: every operator button in one panel (one header instead of one per button): stand, balance, go,
  stop, damp, reset (`/g1_wb/operator`), obstacles virtual / real / both (`/g1_wb/obstacle_mode`), the scenes
  (`/g1_wb/scene`, list from the layout's panel config), and the operator speed slider (`/g1_wb/speed`, only for
  policies trained with an operator speed; the executor refuses it otherwise). The top line shows the executor's
  mode, speed and stop latch from `/g1_wb/status`.

Build: `./pack.sh` -> `g1-wholebody-panels.foxe` (plain JavaScript, no build step). Install: drag the `.foxe` onto
the Foxglove window (or Settings -> Extensions -> install local), then import the layout.
