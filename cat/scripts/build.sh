#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$project_dir/scripts/fetch_runtime.sh"
exec sudo docker build --file "$project_dir/cat/docker/Dockerfile" \
  --tag g1-cat-perception:preview "$project_dir"
