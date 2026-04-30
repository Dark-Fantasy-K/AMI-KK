"""
FRED dataset loader.

Expected on-disk layout:
  FRED/
    train/
      sequences/
        seq_001/
          rgb/          # 000001.jpg …
          events/       # 000001.npy … (x y t p arrays)
          labels/       # 000001.txt  … (YOLO format: class cx cy w h)
        seq_002/
          …
    test/
      sequences/
        …

Each label line: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

If the dataset ships pre-rendered event frames instead of raw events,
set `use_event_frames=True`; then events/ should contain .png images.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.event_processor import load_events, events_to_frame, events_to_voxel


class FREDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        input_size: int = 640,
        num_event_bins: int = 5,
        fusion_mode: str = "late",
        use_event_frames: bool = False,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.num_event_bins = num_event_bins
        self.fusion_mode = fusion_mode
        self.use_event_frames = use_event_frames
        self.augment = augment and (split == "train")

        self.samples: list[dict] = []
        self._collect_samples()

    # ------------------------------------------------------------------
    def _collect_samples(self):
        split_dir = self.root / self.split / "sequences"
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")

        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            rgb_dir = seq_dir / "rgb"
            lbl_dir = seq_dir / "labels"
            evt_dir = seq_dir / "events"

            for rgb_path in sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png")):
                stem = rgb_path.stem
                lbl_path = lbl_dir / f"{stem}.txt"
                if not lbl_path.exists():
                    continue
                if self.use_event_frames:
                    evt_path = evt_dir / f"{stem}.png"
                else:
                    evt_path = evt_dir / f"{stem}.npy"
                self.samples.append({
                    "rgb": str(rgb_path),
                    "evt": str(evt_path) if evt_path.exists() else None,
                    "lbl": str(lbl_path),
                })

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]

        # ---- RGB -------------------------------------------------------
        img = cv2.imread(s["rgb"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]

        # ---- Events → event frame (1 ch) --------------------------------
        if s["evt"] is not None:
            if self.use_event_frames:
                evf = cv2.imread(s["evt"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 127.5 - 1.0
            else:
                evs = load_events(s["evt"])
                evf = events_to_frame(evs, h_orig, w_orig)
        else:
            evf = np.zeros((h_orig, w_orig), dtype=np.float32)

        # ---- Resize to input_size × input_size (letterbox) -------------
        img, evf, scale, pad = self._letterbox(img, evf)

        # ---- Augmentation (training only) ------------------------------
        if self.augment:
            img, evf = self._augment(img, evf)

        # ---- Labels (YOLO format, already normalised) ------------------
        labels = self._load_labels(s["lbl"])
        # labels shape: (N, 5) → [class, cx, cy, w, h]  normalised to original H,W
        # No geometric correction needed since letterbox only adds padding

        # ---- To tensor -------------------------------------------------
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3,H,W)
        evf_t = torch.from_numpy(evf).unsqueeze(0)                                  # (1,H,W)
        x = torch.cat([img_t, evf_t], dim=0)                                        # (4,H,W)

        labels_t = torch.from_numpy(labels).float() if len(labels) > 0 else torch.zeros((0, 5))
        return x, labels_t

    # ------------------------------------------------------------------
    def _letterbox(self, img, evf):
        h, w = img.shape[:2]
        s = self.input_size
        scale = min(s / h, s / w)
        nh, nw = int(h * scale), int(w * scale)
        img_r = cv2.resize(img, (nw, nh))
        evf_r = cv2.resize(evf, (nw, nh))

        pad_top  = (s - nh) // 2
        pad_left = (s - nw) // 2
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        ecanvas = np.zeros((s, s), dtype=np.float32)
        canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = img_r
        ecanvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = evf_r
        return canvas, ecanvas, scale, (pad_left, pad_top)

    def _augment(self, img, evf):
        # Horizontal flip
        if np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            evf = evf[:, ::-1].copy()
        # Brightness jitter
        factor = np.random.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return img, evf

    def _load_labels(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    rows.append([float(v) for v in parts])
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def collate_fn(batch):
    """Custom collate: labels have variable length per image."""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    # prepend image index to each label row
    targets = []
    for i, lbl in enumerate(labels):
        if len(lbl) > 0:
            idx = torch.full((len(lbl), 1), i, dtype=torch.float32)
            targets.append(torch.cat([idx, lbl], dim=1))
    targets = torch.cat(targets, dim=0) if targets else torch.zeros((0, 6))
    return imgs, targets
