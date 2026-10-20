#!/usr/bin/env bash
# validate | run <ros_executor args> | fake <ros_fake_robot args> | sim <sim_loop args> | shell
set -eo pipefail
source /opt/ros/foxy/setup.bash
source /opt/wb/messages/setup.bash
set -u
case "${1:-validate}" in
  validate)
    cd /opt/wb
    python3 -c "import numpy, scipy, skfmm, onnxruntime; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'onnxruntime', onnxruntime.__version__)"
    python3 -c "from unitree_hg.msg import LowCmd, LowState; print('unitree_hg LowCmd/LowState: ok')"
    python3 -m unittest discover -s /opt/wb/cat/tests -p 'test_wholebody.py' -v
    exec python3 -m unittest discover -s /opt/wb/tests -v ;;
  run) shift; exec python3 -m g1_wholebody.ros_executor "$@" ;;
  fake) shift; exec python3 -m g1_wholebody.ros_fake_robot "$@" ;;
  sim) shift; exec python3 -m g1_wholebody.sim_loop "$@" ;;
  shell) shift; exec bash "$@" ;;
  *) echo "usage: validate | run ... | fake ... | sim ... | shell" >&2; exit 2 ;;
esac
