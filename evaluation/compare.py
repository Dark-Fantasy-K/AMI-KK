#!/usr/bin/env python3
"""
End-to-end comparison of three detection pipelines on FRED test set.

Pipelines evaluated:
  1. Hybrid model (RGB + Events) — from Phase 1 training
  2. E2VID → YOLOv8               — reconstruct events, detect on grey frames
  3. FireNet → YOLOv8             — same with FireNet reconstruction

Prints a comparison table to stdout.

Usage:
    python evaluation/compare.py \
        --fred_root      /data/FRED \
        --e2vid_frames   /data/reconstructed/e2vid \
        --firenet_frames /data/reconstructed/firenet \
        [--hybrid_checkpoint /data/checkpoints/best.pt] \
        [--device cuda]

If --hybrid_checkpoint is omitted, the hybrid row shows "N/A (not trained)".
Run the reconstruction scripts first to populate e2vid_frames / firenet_frames.
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hybrid model mAP (Phase 1)
# ---------------------------------------------------------------------------

def _eval_hybrid(
    fred_root: Path,
    checkpoint: Path,
    device: torch.device,
    conf: float,
    iou: float,
) -> dict[str, float]:
    """
    Run the Phase 1 HybridDetector on FRED test set and compute mAP.

    Imports the hybrid model lazily so this script works even without
    training the model (just returns N/A).
    """
    from models.detection.hybrid.model import HybridDetector
    from preprocessing.dataset import FREDDataset, collate_fn
    from torch.utils.data import DataLoader
    from evaluation.detect_on_frames import compute_map, box_iou

    import yaml
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    with open(cfg_path) as f:
        import yaml as _yaml
        cfg = _yaml.safe_load(f)

    ds = FREDDataset(str(fred_root), split="test", augment=False,
                     fusion_mode=cfg["dataset"]["fusion_mode"])
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=collate_fn)

    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = HybridDetector(
        num_classes=cfg["dataset"]["num_classes"],
        fusion_mode=cfg["dataset"]["fusion_mode"],
        pretrained=False,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()

    anchors_cfg = cfg["anchors"]
    anchors = [
        anchors_cfg["small"],
        anchors_cfg["medium"],
        anchors_cfg["large"],
    ]
    input_size = cfg["dataset"]["input_size"]

    all_preds, all_gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            # Decode predictions → boxes (simple argmax per scale)
            decoded = _decode_preds(preds, anchors, input_size, conf)
            all_preds.append(decoded)

            gt_boxes = []
            for row in targets:
                if row[0] == 0:  # image index 0 in this batch
                    _, cx, cy, w, h = row[1:]
                    x1 = (cx - w / 2) * input_size
                    y1 = (cy - h / 2) * input_size
                    x2 = (cx + w / 2) * input_size
                    y2 = (cy + h / 2) * input_size
                    gt_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
            all_gts.append({"boxes": gt_boxes})

    return compute_map(all_preds, all_gts)


def _decode_preds(
    preds: list[torch.Tensor],
    anchors: list,
    input_size: int,
    conf_thr: float,
) -> dict:
    """Minimal decoder: extract high-confidence boxes from YOLO predictions."""
    import numpy as np

    boxes_all, scores_all = [], []
    strides = [8, 16, 32]

    for pred, anch, stride in zip(preds, anchors, strides):
        # pred: (1, na, H, W, 5+nc)
        pred = pred[0]  # (na, H, W, 5+nc)
        na, H, W, _ = pred.shape

        obj   = torch.sigmoid(pred[..., 4])
        mask  = obj > conf_thr
        if not mask.any():
            continue

        for a_i in range(na):
            ys, xs = torch.where(mask[a_i])
            for (y_cell, x_cell) in zip(ys.tolist(), xs.tolist()):
                v = pred[a_i, y_cell, x_cell]
                obj_s = torch.sigmoid(v[4]).item()
                tx, ty, tw, th = v[0].item(), v[1].item(), v[2].item(), v[3].item()
                aw, ah = anch[a_i]
                cx = (x_cell + torch.sigmoid(torch.tensor(tx)).item()) * stride
                cy = (y_cell + torch.sigmoid(torch.tensor(ty)).item()) * stride
                bw = np.exp(tw) * aw
                bh = np.exp(th) * ah
                x1 = cx - bw / 2; y1 = cy - bh / 2
                x2 = cx + bw / 2; y2 = cy + bh / 2
                boxes_all.append([x1, y1, x2, y2])
                scores_all.append(obj_s)

    return {"boxes": boxes_all, "scores": scores_all, "labels": [0] * len(boxes_all)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare three detection pipelines on FRED")
    p.add_argument("--fred_root",         required=True, help="Path to FRED dataset root")
    p.add_argument("--e2vid_frames",      required=True, help="E2VID reconstructed frames dir")
    p.add_argument("--firenet_frames",    required=True, help="FireNet reconstructed frames dir")
    p.add_argument("--hybrid_checkpoint", default=None,  help="Phase 1 checkpoint (optional)")
    p.add_argument("--conf",  type=float, default=0.25)
    p.add_argument("--iou",   type=float, default=0.45)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_table(rows: list[dict]) -> None:
    col_w = {
        "Method":       35,
        "mAP@0.5":      10,
        "mAP@0.5:0.95": 14,
        "Time (s)":     10,
    }
    header = "  ".join(k.ljust(w) for k, w in col_w.items())
    sep    = "  ".join("-" * w for w in col_w.values())
    print()
    print("=" * len(header))
    print("  FRED Detection Comparison")
    print("=" * len(header))
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(
            _fmt(row.get(k, "N/A")).ljust(w)
            for k, w in col_w.items()
        )
        print(line)
    print("=" * len(header))
    print()


def main() -> None:
    args  = parse_args()
    device = torch.device(args.device)

    from evaluation.detect_on_frames import run_detection_pipeline

    fred_root      = Path(args.fred_root)
    e2vid_frames   = Path(args.e2vid_frames)
    firenet_frames = Path(args.firenet_frames)

    rows: list[dict] = []

    # ---- 1. E2VID → YOLOv8 -----------------------------------------------
    log.info("=== Pipeline 1/3: E2VID → YOLOv8 ===")
    t0 = time.time()
    if e2vid_frames.exists():
        m = run_detection_pipeline(e2vid_frames, fred_root, args.conf, args.iou,
                                   device=args.device)
        rows.append({"Method": "E2VID → YOLOv8 (COCO)",
                     **{k: round(v, 4) for k, v in m.items()},
                     "Time (s)": round(time.time() - t0, 1)})
    else:
        log.warning("E2VID frames not found at %s — skipping", e2vid_frames)
        rows.append({"Method": "E2VID → YOLOv8", "mAP@0.5": "N/A",
                     "mAP@0.5:0.95": "N/A", "Time (s)": "N/A"})

    # ---- 2. FireNet → YOLOv8 ---------------------------------------------
    log.info("=== Pipeline 2/3: FireNet → YOLOv8 ===")
    t0 = time.time()
    if firenet_frames.exists():
        m = run_detection_pipeline(firenet_frames, fred_root, args.conf, args.iou,
                                   device=args.device)
        rows.append({"Method": "FireNet → YOLOv8 (COCO)",
                     **{k: round(v, 4) for k, v in m.items()},
                     "Time (s)": round(time.time() - t0, 1)})
    else:
        log.warning("FireNet frames not found at %s — skipping", firenet_frames)
        rows.append({"Method": "FireNet → YOLOv8", "mAP@0.5": "N/A",
                     "mAP@0.5:0.95": "N/A", "Time (s)": "N/A"})

    # ---- 3. Hybrid RGB+Event model (Phase 1) -----------------------------
    log.info("=== Pipeline 3/3: Hybrid RGB+Event detector ===")
    t0 = time.time()
    if args.hybrid_checkpoint and Path(args.hybrid_checkpoint).exists():
        try:
            m = _eval_hybrid(fred_root, Path(args.hybrid_checkpoint), device,
                             args.conf, args.iou)
            rows.append({"Method": "Hybrid (RGB + Event, Phase 1)",
                         **{k: round(v, 4) for k, v in m.items()},
                         "Time (s)": round(time.time() - t0, 1)})
        except Exception as e:
            log.error("Hybrid evaluation failed: %s", e)
            rows.append({"Method": "Hybrid (RGB + Event)", "mAP@0.5": "ERROR",
                         "mAP@0.5:0.95": "ERROR", "Time (s)": "N/A"})
    else:
        log.info("No hybrid checkpoint provided — skipping.")
        rows.append({"Method": "Hybrid (RGB + Event, Phase 1)",
                     "mAP@0.5": "N/A (not trained)",
                     "mAP@0.5:0.95": "N/A",
                     "Time (s)": "N/A"})

    print_table(rows)


if __name__ == "__main__":
    main()
