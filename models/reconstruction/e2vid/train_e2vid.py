#!/usr/bin/env python3
"""
train_e2vid.py — Train (or fine-tune) E2VID on the FRED dataset.

Architecture: recurrent UNet (Rebecq et al., CVPR 2019 / TPAMI 2021)
Input : voxel grid  (num_bins, H, W)
Target: grayscale intensity frame (1, H, W) in [0, 1]
Loss  : λ_l1 · L1 + λ_ssim · (1 − SSIM) + λ_tv · TV

Truncated BPTT is used so each optimizer step covers `tbptt_steps` frames
while the LSTM states persist (detached) across the full window.

Usage — fine-tune from official rpg_e2vid checkpoint:
    python train_e2vid.py \\
        --fred_root  /data/FRED \\
        --weights    models/reconstruction/e2vid/pretrained_weights/E2VID_lightweight.pth.tar \\
        --save_dir   checkpoints/e2vid \\
        --epochs     50

Usage — train from scratch:
    python train_e2vid.py \\
        --fred_root /data/FRED \\
        --save_dir  checkpoints/e2vid \\
        --epochs    100 --lr 2e-4

Resume training:
    python train_e2vid.py ... --resume checkpoints/e2vid/last.pth
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.reconstruction.e2vid.model import E2VID, load_e2vid
from models.reconstruction.utils import (
    events_to_voxel,
    load_events_for_sequence,
    load_frame_timestamps,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class E2VIDTrainDataset(Dataset):
    """
    Sliding-window dataset over FRED sequences.

    Each item is a temporally-contiguous window of K frames:
        voxels  : Tensor (K, num_bins, H, W)  — event voxel grids
        targets : Tensor (K, 1, H, W)         — grayscale RGB frames in [0,1]

    Consecutive windows from the same sequence overlap by window_size//2 so
    every frame appears in multiple windows and the model sees temporal context
    at both boundaries.
    """

    def __init__(
        self,
        fred_root: str | Path,
        split: str = "train",
        num_bins: int = 5,
        window_size: int = 10,
        input_size: int | None = None,
        augment: bool = True,
    ):
        self.num_bins = num_bins
        self.window_size = window_size
        self.input_size = input_size
        self.augment = augment and (split == "train")

        seq_root = Path(fred_root) / split / "sequences"
        if not seq_root.exists():
            raise FileNotFoundError(f"Dataset split not found: {seq_root}")

        # index: list of (seq_dir, start_frame_index)
        self.windows: list[tuple[Path, int]] = []
        self._index(seq_root)

    def _index(self, seq_root: Path) -> None:
        stride = max(1, self.window_size // 2)
        for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
            rgb_dir = seq_dir / "rgb"
            frames = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
            n = len(frames)
            if n < self.window_size:
                log.debug("Skipping %s: only %d frames < window_size=%d", seq_dir.name, n, self.window_size)
                continue
            for start in range(0, n - self.window_size + 1, stride):
                self.windows.append((seq_dir, start))

        if not self.windows:
            raise RuntimeError(
                "No training windows found. Check --fred_root and --window_size."
            )
        log.info("Indexed %d windows from %s", len(self.windows), seq_root)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_dir, start = self.windows[idx]

        events = load_events_for_sequence(seq_dir)
        rgb_paths = sorted((seq_dir / "rgb").glob("*.jpg")) + \
                    sorted((seq_dir / "rgb").glob("*.png"))
        timestamps = load_frame_timestamps(seq_dir)
        if len(timestamps) != len(rgb_paths):
            timestamps = np.linspace(0.0, (len(rgb_paths) - 1) / 30.0, len(rgb_paths))

        # Reference frame size
        ref = cv2.imread(str(rgb_paths[start]))
        H_src, W_src = ref.shape[:2]
        H = W = self.input_size if self.input_size else None

        flip = self.augment and random.random() < 0.5

        voxels: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []

        for fi in range(start, start + self.window_size):
            # ── Target frame ──────────────────────────────────────────────
            img_bgr = cv2.imread(str(rgb_paths[fi]))
            if img_bgr is None:
                img_bgr = np.zeros((H_src, W_src, 3), dtype=np.uint8)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # (H, W) uint8

            # ── Voxel grid ────────────────────────────────────────────────
            t_end   = timestamps[fi]
            t_start = timestamps[fi - 1] if fi > 0 else t_end - 1.0 / 30.0
            voxel   = events_to_voxel(events, H_src, W_src, self.num_bins, t_start, t_end)

            # ── Resize if requested ───────────────────────────────────────
            if self.input_size:
                gray  = cv2.resize(gray,  (self.input_size, self.input_size))
                voxel = np.stack([
                    cv2.resize(voxel[b], (self.input_size, self.input_size))
                    for b in range(self.num_bins)
                ])

            # ── Augmentation ──────────────────────────────────────────────
            if flip:
                gray  = gray[:,  ::-1].copy()
                voxel = voxel[:, :, ::-1].copy()

            voxels.append(torch.from_numpy(voxel.copy()))
            targets.append(
                torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)
            )

        return torch.stack(voxels), torch.stack(targets)
        # shapes: (K, num_bins, H, W),  (K, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def _ssim(pred: torch.Tensor, target: torch.Tensor, k: int = 11) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = k // 2

    def pool(x):
        return F.avg_pool2d(x, k, stride=1, padding=pad)

    mu_p, mu_t = pool(pred), pool(target)
    sigma_p  = pool(pred   * pred)   - mu_p * mu_p
    sigma_t  = pool(target * target) - mu_t * mu_t
    sigma_pt = pool(pred   * target) - mu_p * mu_t

    num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2)
    return (num / den).mean()


def _tv(x: torch.Tensor) -> torch.Tensor:
    return (
        (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    )


class E2VIDLoss(nn.Module):
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_tv: float = 1e-4,
    ):
        super().__init__()
        self.w_l1   = lambda_l1
        self.w_ssim = lambda_ssim
        self.w_tv   = lambda_tv

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l1   = F.l1_loss(pred, target)
        ssim = 1.0 - _ssim(pred, target)
        tv   = _tv(pred)
        loss = self.w_l1 * l1 + self.w_ssim * ssim + self.w_tv * tv
        return loss, {"l1": l1.item(), "ssim": ssim.item(), "tv": tv.item()}


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate one epoch
# ─────────────────────────────────────────────────────────────────────────────

def _detach_states(model: E2VID) -> None:
    """Detach ConvLSTM states in-place to stop gradient flow across TBPTT chunks."""
    for i, s in enumerate(model.unet.states):
        if s is not None:
            h, c = s
            model.unet.states[i] = (h.detach(), c.detach())


def train_epoch(
    model: E2VID,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: E2VIDLoss,
    device: torch.device,
    tbptt_steps: int = 5,
    grad_clip: float = 1.0,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
    n = 0

    for voxels, targets in loader:
        # voxels : (B, K, num_bins, H, W)
        # targets: (B, K, 1, H, W)
        voxels  = voxels.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        K = voxels.shape[1]
        model.unet.reset_states()

        for t0 in range(0, K, tbptt_steps):
            t1 = min(t0 + tbptt_steps, K)
            optimizer.zero_grad()
            chunk_loss = torch.tensor(0.0, device=device)

            for t in range(t0, t1):
                with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                    pred = model(voxels[:, t])
                    loss, parts = criterion(pred, targets[:, t])
                chunk_loss = chunk_loss + loss

            if scaler is not None:
                scaler.scale(chunk_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                chunk_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            _detach_states(model)

            totals["loss"] += chunk_loss.item() / (t1 - t0)
            totals["l1"]   += parts["l1"]
            totals["ssim"] += parts["ssim"]
            n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: E2VID,
    loader: DataLoader,
    criterion: E2VIDLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
    n = 0

    for voxels, targets in loader:
        voxels  = voxels.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        K = voxels.shape[1]
        model.unet.reset_states()

        for t in range(K):
            pred = model(voxels[:, t])
            loss, parts = criterion(pred, targets[:, t])
            totals["loss"] += loss.item()
            totals["l1"]   += parts["l1"]
            totals["ssim"] += parts["ssim"]
            n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Warmup scheduler wrapper
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, eta_min_ratio: float = 0.01):
        import math
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
        super().__init__(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train E2VID on the FRED event-camera dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--fred_root",   default="/data/FRED",        help="FRED dataset root directory")
    p.add_argument("--input_size",  type=int, default=None,      help="Resize frames to N×N; None = original size")
    p.add_argument("--num_bins",    type=int, default=5,          help="Voxel grid time bins")
    p.add_argument("--window_size", type=int, default=10,         help="Frames per training window")
    p.add_argument("--num_workers", type=int, default=4)
    # Model
    p.add_argument("--weights",  default=None, help="Pretrained rpg_e2vid checkpoint for fine-tuning")
    p.add_argument("--resume",   default=None, help="Resume from our checkpoint (last.pth)")
    # Training
    p.add_argument("--save_dir",     default="checkpoints/e2vid")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup",       type=int,   default=3,       help="LR warmup epochs")
    p.add_argument("--tbptt",        type=int,   default=5,       help="Truncated-BPTT chunk size (frames)")
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--amp",          action="store_true",          help="Mixed-precision training (CUDA only)")
    # Loss weights
    p.add_argument("--lambda_l1",   type=float, default=1.0)
    p.add_argument("--lambda_ssim", type=float, default=0.5)
    p.add_argument("--lambda_tv",   type=float, default=1e-4)
    # Misc
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed",   type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = E2VIDTrainDataset(
        args.fred_root, split="train",
        num_bins=args.num_bins, window_size=args.window_size,
        input_size=args.input_size, augment=True,
    )
    val_ds = E2VIDTrainDataset(
        args.fred_root, split="test",
        num_bins=args.num_bins, window_size=args.window_size,
        input_size=args.input_size, augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    log.info("Train: %d windows  |  Val: %d windows", len(train_ds), len(val_ds))

    # ── Model ─────────────────────────────────────────────────────────────
    if args.weights:
        model = load_e2vid(args.weights, device)
        model.train()
        log.info("Fine-tuning from %s", args.weights)
    else:
        model = E2VID(num_bins=args.num_bins).to(device)
        log.info("Training from scratch")

    # ── Optimizer / scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineScheduler(optimizer, args.warmup, args.epochs)
    criterion = E2VIDLoss(args.lambda_l1, args.lambda_ssim, args.lambda_tv)
    scaler    = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        log.info("Resumed from epoch %d  (best val=%.4f)", start_epoch, best_val_loss)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = train_epoch(
            model, train_loader, optimizer, criterion, device,
            tbptt_steps=args.tbptt, grad_clip=args.grad_clip, scaler=scaler,
        )
        vl = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        log.info(
            "Epoch %3d/%d | %4.0fs | lr=%.2e | "
            "train: loss=%.4f L1=%.4f SSIM=%.4f | "
            "val: loss=%.4f L1=%.4f",
            epoch + 1, args.epochs, elapsed,
            scheduler.get_last_lr()[0],
            tr["loss"], tr["l1"], tr["ssim"],
            vl["loss"], vl["l1"],
        )

        ckpt_data = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args":          vars(args),
        }
        torch.save(ckpt_data, save_dir / "last.pth")

        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            torch.save(ckpt_data, save_dir / "best.pth")
            log.info("  ✓ best val loss → %.4f", best_val_loss)

    log.info("Done. Best val loss: %.4f  Checkpoints: %s", best_val_loss, save_dir)


if __name__ == "__main__":
    main()
