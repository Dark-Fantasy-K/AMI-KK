#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-pretrained_weights}"
mkdir -p "$OUTDIR"

OUTFILE="$OUTDIR/E2VID_lightweight.pth.tar"
URL="http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar"

if [ -f "$OUTFILE" ]; then
    echo "Weights already present: $OUTFILE"
    exit 0
fi

echo "Downloading E2VID weights..."
wget "$URL" -O "$OUTFILE"

echo "Done: $OUTFILE"