#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset=g1-foxy-arm64-runtime-20260908.tar.gz
url="https://github.com/Skvayzer/humanoid_navigation/releases/download/robot-snapshot-20260908/$asset"
mkdir -p "$project_dir/artifacts"
archive="$project_dir/artifacts/$asset"
if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 "$url" --output "$archive.part"
  mv "$archive.part" "$archive"
fi
(cd "$project_dir/artifacts" && sha256sum -c ../manifests/runtime_asset.sha256)
# Preserve any existing local runtime modifications instead of overwriting.
if (cd "$project_dir" && sha256sum --quiet -c manifests/runtime_files.sha256 >/dev/null 2>&1); then
  echo 'Tested runtime is already present and verified.'
  exit 0
fi
for target in "$project_dir/vendor/open3d" "$project_dir/vendor/unitree_cyclone/libddsc.so" "$project_dir/vendor/unitree_cyclone/librmw_cyclonedds_cpp.so"; do
  if [[ -e "$target" || -L "$target" ]]; then
    echo "Existing runtime differs from snapshot; refusing overwrite: $target" >&2
    exit 3
  fi
done
tar --extract --gzip --file "$archive" --directory "$project_dir" --no-same-owner --keep-old-files
(cd "$project_dir" && sha256sum --quiet -c manifests/runtime_files.sha256)
echo 'Tested ARM64 runtime restored. No host libraries or robot processes changed.'
