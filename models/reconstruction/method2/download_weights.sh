#!/usr/bin/env bash
# Download FireNet pretrained weights.
#
# Source: cedric-scheerlinck/rpg_e2vid (firenet branch)
#   https://github.com/cedric-scheerlinck/rpg_e2vid
#
# If the Google Drive link below expires, get the updated ID from:
#   https://github.com/cedric-scheerlinck/rpg_e2vid#pretrained-models
#
# Requires: gdown (pip install gdown)

set -euo pipefail

OUTDIR="${1:-/data/weights}"
mkdir -p "$OUTDIR"

# FireNet checkpoint — Google Drive file ID
# (update if link expired; check cedric-scheerlinck/rpg_e2vid README)
GDRIVE_ID="1VoMtGDkckk-R4TbD8Bc2UdYkdz-BOjfV"
OUTFILE="$OUTDIR/firenet.pth"

if [ -f "$OUTFILE" ]; then
    echo "Weights already present: $OUTFILE"
    exit 0
fi

if ! command -v gdown &> /dev/null; then
    echo "gdown not found.  Install with:  pip install gdown"
    exit 1
fi

echo "Downloading FireNet weights → $OUTFILE"
gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "$OUTFILE"

echo "Done: $OUTFILE"
