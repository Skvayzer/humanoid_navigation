#!/usr/bin/env bash
# Packs the extension: a .foxe is a zip of package.json + dist/extension.js (+ README).
set -euo pipefail
cd "$(dirname "$0")"
node --check dist/extension.js
rm -f g1-wholebody-panels.foxe
zip -q -r g1-wholebody-panels.foxe package.json README.md dist/extension.js
echo "wrote $(pwd)/g1-wholebody-panels.foxe"
