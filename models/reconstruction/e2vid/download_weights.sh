#!/usr/bin/env bash
# Download official E2VID pretrained weights from rpg_e2vid.
#
# Weights page: https://github.com/uzh-rpg/rpg_e2vid#pretrained-models
# Download requires gdown (pip install gdown).
#
# The official README lists two checkpoints:
#   E2VID       — trained on synthetic + real data
#   E2VID_fire  — lighter FireNet-style variant
#
# Update GDRIVE_ID below if the link has changed (check the GitHub README).

set -euo pipefail

OUTDIR="${1:-/data/weights}"
mkdir -p "$OUTDIR"

# Official E2VID checkpoint (rpg_e2vid TPAMI 2021 version)
# Google Drive file ID — verify at https://github.com/uzh-rpg/rpg_e2vid
GDRIVE_ID="1dbesB3KGsWpn0kYEj8jLrfTBJJhZBqc1"
OUTFILE="$OUTDIR/e2vid.pth"

if [ -f "$OUTFILE" ]; then
    echo "Weights already present: $OUTFILE"
    exit 0
fi

if ! command -v gdown &> /dev/null; then
    echo "gdown not found.  Install with:  pip install gdown"
    exit 1
fi

echo "Downloading E2VID weights → $OUTFILE"
gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "$OUTFILE"

echo "Done: $OUTFILE"
