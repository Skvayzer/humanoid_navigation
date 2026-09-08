#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
destination="${G1_BIN_DIR:-$HOME/bin}"
# Preflight everything before creating anything. Existing commands are not ours
# to replace, even when they happen to have the same names.
for command in "$G1_SLAM_PROJECT_DIR"/host/bin/*; do
  target="$destination/$(basename "$command")"
  if [[ -e "$target" || -L "$target" ]]; then
    if [[ "$(readlink -f "$target")" != "$command" ]]; then
      echo "Refusing to overwrite existing command: $target" >&2
      echo 'Use a separate G1_BIN_DIR and an explicit PATH for this checkout.' >&2
      exit 3
    fi
  fi
done
mkdir -p "$destination" "$G1_SLAM_RUNTIME_DIR/maps" "$G1_SLAM_RUNTIME_DIR/nav" "$G1_SLAM_RUNTIME_DIR/logs/ros"
for command in "$G1_SLAM_PROJECT_DIR"/host/bin/*; do
  target="$destination/$(basename "$command")"
  [[ -L "$target" ]] || ln -s "$command" "$target"
done
printf 'Shortcuts installed in %s. No service was started.\n' "$destination"
printf 'For this shell: export PATH="%s:$PATH"\n' "$destination"
