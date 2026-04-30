"""
Shared utilities for reconstruction modules:
  - load events from FRED sequence
  - build voxel grid for each frame timestamp
  - save output frames as PNG
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Event loading helpers
# ---------------------------------------------------------------------------

def load_events_for_sequence(seq_dir: Path) -> dict:
    """
    Load ALL events for a sequence into memory.

    Searches for events.npy / events.h5 / events.txt in seq_dir/events/.
    Returns dict with arrays: x (int32), y (int32), t (float64), p (int8).
    """
    evt_dir = seq_dir / "events"

    for name, loader in [
        ("events.npy", _load_npy),
        ("events.h5",  _load_h5),
        ("events.txt", _load_txt),
    ]:
        path = evt_dir / name
        if path.exists():
            return loader(path)

    # Fallback: per-frame npy files → concat
    npy_files = sorted(evt_dir.glob("*.npy"))
    if npy_files:
        chunks = [_load_npy(p) for p in npy_files]
        return {k: np.concatenate([c[k] for c in chunks]) for k in ("x", "y", "t", "p")}

    raise FileNotFoundError(f"No event data found in {evt_dir}")


def _load_npy(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    if data.dtype.names:
        return {k: data[k] for k in ("x", "y", "t", "p")}
    assert data.shape[1] == 4
    return {"x": data[:, 0].astype(np.int32),
            "y": data[:, 1].astype(np.int32),
            "t": data[:, 2].astype(np.float64),
            "p": data[:, 3].astype(np.int8)}


def _load_h5(path: Path) -> dict:
    import h5py
    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in ("x", "y", "t", "p")}


def _load_txt(path: Path) -> dict:
    data = np.loadtxt(path)
    return {"x": data[:, 0].astype(np.int32),
            "y": data[:, 1].astype(np.int32),
            "t": data[:, 2].astype(np.float64),
            "p": data[:, 3].astype(np.int8)}


# ---------------------------------------------------------------------------
# Frame timestamps
# ---------------------------------------------------------------------------

def load_frame_timestamps(seq_dir: Path) -> np.ndarray:
    """
    Return sorted array of timestamps (float64) at which RGB frames were captured.

    Tries (in order):
      1. timestamps.txt  (one float per line, or 'index timestamp' format)
      2. timestamps.json
      3. Infer from number of RGB frames at 30 fps (fallback).
    """
    ts_txt = seq_dir / "timestamps.txt"
    ts_json = seq_dir / "timestamps.json"

    if ts_txt.exists():
        data = np.loadtxt(ts_txt)
        if data.ndim == 2:          # two-column: index timestamp
            data = data[:, 1]
        return data.astype(np.float64)

    if ts_json.exists():
        with open(ts_json) as f:
            return np.array(json.load(f), dtype=np.float64)

    # Fallback: infer from RGB frame count at 30 fps
    rgb_frames = sorted((seq_dir / "rgb").glob("*.jpg")) + \
                 sorted((seq_dir / "rgb").glob("*.png"))
    n = len(rgb_frames)
    if n == 0:
        raise FileNotFoundError(f"No RGB frames in {seq_dir}/rgb/")
    return np.linspace(0.0, (n - 1) / 30.0, n)


# ---------------------------------------------------------------------------
# Voxel grid construction
# ---------------------------------------------------------------------------

def events_to_voxel(
    events: dict,
    height: int,
    width: int,
    num_bins: int,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """
    Build a (num_bins, H, W) voxel grid from events in [t_start, t_end).

    Uses linear interpolation across time bins (standard approach).
    Returns float32 array normalised per-bin to [-1, 1].
    """
    x, y, t, p = (events[k] for k in ("x", "y", "t", "p"))

    mask = (t >= t_start) & (t < t_end)
    x, y, t, p = x[mask], y[mask], t[mask], p[mask]

    # Clip to valid pixel range
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]

    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if len(x) == 0:
        return voxel

    dt = t_end - t_start
    if dt <= 0:
        return voxel

    t_norm = (t - t_start) / dt * (num_bins - 1)   # in [0, num_bins-1]
    pol = p.astype(np.float32) * 2 - 1              # {0,1} → {-1,+1}

    tb = np.clip(t_norm.astype(np.int32), 0, num_bins - 2)
    alpha = t_norm - tb

    np.add.at(voxel, (tb,     y, x), (1.0 - alpha) * pol)
    np.add.at(voxel, (tb + 1, y, x),         alpha  * pol)

    for b in range(num_bins):
        m = np.abs(voxel[b]).max()
        if m > 0:
            voxel[b] /= m

    return voxel


# ---------------------------------------------------------------------------
# Frame saving / loading
# ---------------------------------------------------------------------------

def save_frame(img: np.ndarray, path: Path, as_color: bool = False) -> None:
    """Save a float32/uint8 image (H,W) or (H,W,3) as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if as_color and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), img)


def frame_iter(
    seq_dir: Path,
    events: dict,
    height: int,
    width: int,
    num_bins: int,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Iterate over (stem, rgb_img, voxel_grid) for every frame in a sequence.

    stem:       filename stem (e.g. '000001')
    rgb_img:    (H, W, 3) uint8 BGR
    voxel_grid: (num_bins, H, W) float32
    """
    rgb_paths = sorted((seq_dir / "rgb").glob("*.jpg")) + \
                sorted((seq_dir / "rgb").glob("*.png"))
    timestamps = load_frame_timestamps(seq_dir)

    if len(timestamps) != len(rgb_paths):
        # Recompute at 30 fps
        timestamps = np.linspace(0.0, (len(rgb_paths) - 1) / 30.0, len(rgb_paths))

    for i, rgb_path in enumerate(rgb_paths):
        img = cv2.imread(str(rgb_path))
        if img is None:
            continue

        t_end = timestamps[i]
        t_start = timestamps[i - 1] if i > 0 else t_end - 1.0 / 30.0

        voxel = events_to_voxel(events, height, width, num_bins, t_start, t_end)
        yield rgb_path.stem, img, voxel


# ---------------------------------------------------------------------------
# FRED sequence discovery
# ---------------------------------------------------------------------------

def find_test_sequences(fred_root: Path) -> list[Path]:
    """Return list of sequence dirs under fred_root/test/sequences/."""
    seq_root = fred_root / "test" / "sequences"
    if not seq_root.exists():
        raise FileNotFoundError(f"Test split not found: {seq_root}")
    seqs = sorted(p for p in seq_root.iterdir() if p.is_dir())
    if not seqs:
        raise FileNotFoundError(f"No sequences found in {seq_root}")
    return seqs


def get_frame_size(seq_dir: Path) -> Tuple[int, int]:
    """Return (height, width) by reading the first RGB frame."""
    for ext in ("*.jpg", "*.png"):
        frames = sorted((seq_dir / "rgb").glob(ext))
        if frames:
            img = cv2.imread(str(frames[0]))
            if img is not None:
                return img.shape[0], img.shape[1]
    raise FileNotFoundError(f"Cannot determine frame size for {seq_dir}")
