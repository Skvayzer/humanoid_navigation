#!/usr/bin/env bash
set -eo pipefail
# Foxy's generated setup files do not support nounset.
source /opt/ros/foxy/setup.bash
set -u
case "${1:-validate}" in
  validate)
    python3 -c 'import preview_node; print("ROS imports OK; node NOT started")'
    # The launcher runs validate with --network none. Exercise actual Foxy
    # subscription resolution too; imports/mocked lookups cannot catch it.
    export CAT_RUN_ROS_GRAPH_TESTS=1 ROS_DOMAIN_ID=101
    exec python3 -m unittest discover -s /opt/cat/tests -v
    ;;
  preview)
    exec python3 /opt/cat/preview_node.py --ros-args --params-file /opt/cat/config/preview.yaml
    ;;
  *) echo 'Only validate or preview is supported. No motion mode exists.' >&2; exit 2 ;;
esac
