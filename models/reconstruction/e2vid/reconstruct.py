#!/usr/bin/env python3
"""
E2VID reconstruction from a single events file (.npy or .txt).

File format (one event per line):  t x y p
  t  - timestamp (seconds, float)
  x  - x-coordinate (int)
  y  - y-coordinate (int)
  p  - polarity (-1 or +1)

Usage:
    python models/reconstruction/e2vid/reconstruct.py \
        --events_file data/events_v2id.npy \
        --weights     models/reconstruction/e2vid/pretrained_weights/E2VID_lightweight.pth.tar \
        --output      data/reconstructed \
        [--events_per_frame 200000] [--num_bins 5]
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.reconstruction.e2vid.model import load_e2vid
from models.reconstruction.utils import events_to_voxel, save_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_events(path: Path) -> dict:
    """Load events from .npy (fast) or .txt (slow, auto-caches to .npy)."""
    path = Path(path)

    if path.suffix == ".npy":
        log.info("Loading events from %s …", path)
        data = np.load(path)
    else:
        npy_path = path.with_suffix(".npy")
        if npy_path.exists():
            log.info("Loading cached events from %s …", npy_path)
            data = np.load(npy_path)
        else:
            log.info("Loading events from %s (slow, will cache) …", path)
            df = pd.read_csv(path, sep=" ", header=None, names=["t", "x", "y", "p"], engine="c")
            data = df.values
            np.save(npy_path, data)
            log.info("Cached to %s", npy_path)

    return {
        "t": data[:, 0].astype(np.float64),
        "x": data[:, 1].astype(np.int32),
        "y": data[:, 2].astype(np.int32),
        "p": ((data[:, 3] + 1) // 2).astype(np.int8),  # {-1,+1} → {0,1}
    }


@torch.inference_mode()
def reconstruct(
    model,
    events: dict,
    height: int,
    width: int,
    out_dir: Path,
    num_bins: int,
    events_per_frame: int,
    device: torch.device,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Run E2VID frame-by-frame on events dict.
    progress_cb(done, total) is called after each frame if provided.
    Returns number of frames written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.reset_states()

    n = len(events["t"])
    n_frames = (n + events_per_frame - 1) // events_per_frame
    log.info("Total events: %d → %d frames (chunk size %d)", n, n_frames, events_per_frame)

    for i in range(n_frames):
        start = i * events_per_frame
        end = min(start + events_per_frame, n)
        chunk = {k: v[start:end] for k, v in events.items()}

        t_start = float(chunk["t"][0])
        t_end = float(chunk["t"][-1])
        if t_end <= t_start:
            t_end = t_start + 1e-6

        voxel = events_to_voxel(chunk, height, width, num_bins, t_start, t_end)
        x = torch.from_numpy(voxel).unsqueeze(0).to(device)
        pred = model(x)
        img = pred[0, 0].cpu().numpy()

        save_frame(img, out_dir / f"{i:06d}.png")

        if progress_cb:
            progress_cb(i + 1, n_frames)
        elif (i + 1) % 20 == 0:
            log.info("  %d / %d frames", i + 1, n_frames)

    log.info("Done. %d frames → %s", n_frames, out_dir)
    return n_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2VID reconstruction from events file")
    p.add_argument("--events_file",      required=True, help="Path to .npy or .txt events file")
    p.add_argument("--weights",          required=True, help="Path to E2VID .pth checkpoint")
    p.add_argument("--output",           required=True, help="Output directory for frames")
    p.add_argument("--events_per_frame", type=int, default=200000)
    p.add_argument("--num_bins",         type=int, default=5)
    p.add_argument("--height",           type=int, default=None)
    p.add_argument("--width",            type=int, default=None)
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(os.cpu_count() or 1)

    events = load_events(Path(args.events_file))
    height = args.height or int(events["y"].max()) + 1
    width  = args.width  or int(events["x"].max()) + 1
    log.info("Sensor size: %d × %d (H × W)", height, width)

    log.info("Loading E2VID weights from %s", args.weights)
    model = load_e2vid(args.weights, device)

    reconstruct(
        model, events, height, width,
        out_dir=Path(args.output),
        num_bins=args.num_bins,
        events_per_frame=args.events_per_frame,
        device=device,
    )


if __name__ == "__main__":
    main()
