#!/usr/bin/env python3
"""
E2VID reconstruction on FRED test sequences.

Usage:
    python models/reconstruction/e2vid/reconstruct.py \
        --fred_root /data/FRED \
        --weights   /data/weights/e2vid.pth \
        --output    /data/reconstructed/e2vid \
        [--num_bins 5] [--device cuda]

Output layout:
    <output>/
        seq_001/
            000001.png
            000002.png
            …
        seq_002/
            …
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.reconstruction.e2vid.model import load_e2vid
from models.reconstruction.utils import (
    find_test_sequences,
    get_frame_size,
    load_events_for_sequence,
    frame_iter,
    save_frame,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def reconstruct_sequence(
    model,
    seq_dir: Path,
    out_seq_dir: Path,
    num_bins: int,
    device: torch.device,
    save_every: int = 1,
) -> int:
    """
    Run E2VID frame-by-frame on one sequence.

    Returns number of frames written.
    """
    out_seq_dir.mkdir(parents=True, exist_ok=True)

    height, width = get_frame_size(seq_dir)
    events = load_events_for_sequence(seq_dir)

    model.reset_states()
    written = 0

    for stem, rgb_bgr, voxel in frame_iter(seq_dir, events, height, width, num_bins):
        # voxel: (num_bins, H, W) float32
        x = torch.from_numpy(voxel).unsqueeze(0).to(device)   # (1, B, H, W)
        pred = model(x)                                         # (1, 1, H, W)
        img  = pred[0, 0].cpu().numpy()                        # (H, W) in [0,1]

        if written % save_every == 0:
            save_frame(img, out_seq_dir / f"{stem}.png")

        written += 1

    log.info("  %s: %d frames written to %s", seq_dir.name, written, out_seq_dir)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2VID reconstruction on FRED test set")
    p.add_argument("--fred_root", required=True, help="Path to FRED dataset root")
    p.add_argument("--weights",   required=True, help="Path to E2VID .pth checkpoint")
    p.add_argument("--output",    required=True, help="Output directory for reconstructed frames")
    p.add_argument("--num_bins",  type=int, default=5, help="Voxel grid time bins (default 5)")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_every", type=int, default=1,
                   help="Save every N-th frame (1 = all frames, useful to reduce disk usage)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    log.info("Loading E2VID weights from %s", args.weights)
    model = load_e2vid(args.weights, device)

    fred_root = Path(args.fred_root)
    out_root  = Path(args.output)
    sequences = find_test_sequences(fred_root)
    log.info("Found %d test sequences in %s", len(sequences), fred_root)

    total = 0
    for seq_dir in sequences:
        out_seq_dir = out_root / seq_dir.name
        log.info("Processing %s …", seq_dir.name)
        total += reconstruct_sequence(
            model, seq_dir, out_seq_dir,
            num_bins=args.num_bins,
            device=device,
            save_every=args.save_every,
        )

    log.info("Done. Total frames written: %d → %s", total, out_root)


if __name__ == "__main__":
    main()
