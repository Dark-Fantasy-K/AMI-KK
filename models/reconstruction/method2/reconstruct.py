#!/usr/bin/env python3
"""
FireNet reconstruction on FRED test sequences.

Usage:
    python models/reconstruction/method2/reconstruct.py \
        --fred_root /data/FRED \
        --weights   /data/weights/firenet.pth \
        --output    /data/reconstructed/firenet \
        [--num_bins 5] [--device cuda]

Output layout mirrors E2VID:
    <output>/
        seq_001/
            000001.png
            …
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.reconstruction.method2.model import load_firenet
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
# Inference
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
    out_seq_dir.mkdir(parents=True, exist_ok=True)

    height, width = get_frame_size(seq_dir)
    events = load_events_for_sequence(seq_dir)

    model.reset_states()
    written = 0

    for stem, _rgb_bgr, voxel in frame_iter(seq_dir, events, height, width, num_bins):
        x    = torch.from_numpy(voxel).unsqueeze(0).to(device)  # (1, B, H, W)
        pred = model(x)                                           # (1, 1, H, W)
        img  = pred[0, 0].cpu().numpy()                          # (H, W) in [0,1]

        if written % save_every == 0:
            save_frame(img, out_seq_dir / f"{stem}.png")

        written += 1

    log.info("  %s: %d frames → %s", seq_dir.name, written, out_seq_dir)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FireNet reconstruction on FRED test set")
    p.add_argument("--fred_root", required=True)
    p.add_argument("--weights",   required=True)
    p.add_argument("--output",    required=True)
    p.add_argument("--num_bins",  type=int, default=5)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_every", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    log.info("Loading FireNet weights from %s", args.weights)
    model = load_firenet(args.weights, device)

    fred_root = Path(args.fred_root)
    out_root  = Path(args.output)
    sequences = find_test_sequences(fred_root)
    log.info("Found %d test sequences", len(sequences))

    total = 0
    for seq_dir in sequences:
        log.info("Processing %s …", seq_dir.name)
        total += reconstruct_sequence(
            model, seq_dir, out_root / seq_dir.name,
            num_bins=args.num_bins,
            device=device,
            save_every=args.save_every,
        )

    log.info("Done. Total frames: %d → %s", total, out_root)


if __name__ == "__main__":
    main()
