"""
Event camera data → 2-D or voxel representations.

Supported input formats:
  - numpy structured array with fields x, y, t, p   (FRED .npy)
  - HDF5 file with datasets x, y, t, p              (common format)
  - plain dict / namedtuple with arrays x, y, t, p
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional


def load_events(path: str | Path) -> dict:
    """Load events from file → dict with arrays x, y, t, p."""
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
        if data.dtype.names:                              # structured array
            return {k: data[k] for k in ("x", "y", "t", "p")}
        else:                                             # (N,4) array
            assert data.shape[1] == 4, "Expected columns: x y t p"
            return {"x": data[:, 0].astype(np.int32),
                    "y": data[:, 1].astype(np.int32),
                    "t": data[:, 2].astype(np.float64),
                    "p": data[:, 3].astype(np.int8)}
    elif path.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            return {k: f[k][:] for k in ("x", "y", "t", "p")}
    elif path.suffix == ".txt":
        data = np.loadtxt(path)
        return {"x": data[:, 0].astype(np.int32),
                "y": data[:, 1].astype(np.int32),
                "t": data[:, 2].astype(np.float64),
                "p": data[:, 3].astype(np.int8)}
    else:
        raise ValueError(f"Unsupported event file format: {path.suffix}")


def events_to_frame(
    events: dict,
    height: int,
    width: int,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> np.ndarray:
    """Accumulate events into a single polarity image in [-1, 1].

    Returns (H, W) float32.
    """
    frame = np.zeros((height, width), dtype=np.float32)
    x, y, t, p = events["x"], events["y"], events["t"], events["p"]

    mask = np.ones(len(x), dtype=bool)
    if t_start is not None:
        mask &= t >= t_start
    if t_end is not None:
        mask &= t < t_end

    x, y, p = x[mask], y[mask], p[mask]

    # clip to valid range
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p = x[valid], y[valid], p[valid]

    polarity = p.astype(np.float32) * 2 - 1   # {0,1} → {-1,+1}
    np.add.at(frame, (y, x), polarity)

    # normalise to [-1, 1]
    max_val = max(np.abs(frame).max(), 1.0)
    frame /= max_val
    return frame


def events_to_voxel(
    events: dict,
    height: int,
    width: int,
    num_bins: int = 5,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> np.ndarray:
    """Convert events to a voxel grid with linear time interpolation.

    Returns (num_bins, H, W) float32.
    """
    x, y, t, p = events["x"], events["y"], events["t"], events["p"]

    mask = np.ones(len(x), dtype=bool)
    if t_start is not None:
        mask &= t >= t_start
    if t_end is not None:
        mask &= t < t_end

    x, y, t, p = x[mask], y[mask], t[mask], p[mask]

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]

    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if len(x) == 0:
        return voxel

    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        t_norm = np.zeros_like(t, dtype=np.float32)
    else:
        t_norm = (t - t_min) / (t_max - t_min) * (num_bins - 1)  # in [0, B-1]

    polarity = p.astype(np.float32) * 2 - 1

    # Bilinear interpolation across bins
    tb = t_norm.astype(np.int32)
    tb = np.clip(tb, 0, num_bins - 2)
    alpha = t_norm - tb

    np.add.at(voxel, (tb,     y, x), (1 - alpha) * polarity)
    np.add.at(voxel, (tb + 1, y, x),      alpha  * polarity)

    # per-bin normalisation
    for b in range(num_bins):
        m = np.abs(voxel[b]).max()
        if m > 0:
            voxel[b] /= m

    return voxel
