#!/usr/bin/env bash
# Optional offline/unit-test build using already installed development libraries.
# No apt/pip installation, system writes, or ROS node is started here.
set -euo pipefail
cat_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$cat_dir/native" -B "$cat_dir/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$cat_dir/build" --parallel 2
cd "$cat_dir"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -v
