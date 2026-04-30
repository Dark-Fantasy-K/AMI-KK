"""
Run YOLOv8 (ultralytics, pretrained on COCO) on a directory of reconstructed
grayscale frames and compute mAP against FRED ground-truth labels.

Used by evaluation/compare.py to score E2VID→detect and FireNet→detect pipelines.

API:
    from evaluation.detect_on_frames import run_detection_pipeline

    map50, map50_95 = run_detection_pipeline(
        frames_root = Path("/data/reconstructed/e2vid"),
        labels_root = Path("/data/FRED/test"),      # FRED test split root
        conf        = 0.25,
        iou         = 0.45,
        drone_coco_ids = [0],   # COCO class ids to treat as 'drone' (person=0 is a proxy)
    )
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

# FRED labels → class 0 is drone.  For COCO-pretrained YOLO we map detections
# from any COCO class to "drone" as a simplification (or use a fine-tuned model).
# Override with --drone_coco_ids if needed.
DEFAULT_DRONE_COCO_IDS = list(range(80))   # accept any class (maximises recall)


# ---------------------------------------------------------------------------
# mAP helpers
# ---------------------------------------------------------------------------

def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.

    box_a: (N, 4) in [x1 y1 x2 y2]
    box_b: (M, 4)
    Returns: (N, M) IoU matrix
    """
    xa = np.maximum(box_a[:, None, 0], box_b[None, :, 0])
    ya = np.maximum(box_a[:, None, 1], box_b[None, :, 1])
    xb = np.minimum(box_a[:, None, 2], box_b[None, :, 2])
    yb = np.minimum(box_a[:, None, 3], box_b[None, :, 3])

    inter = np.maximum(xb - xa, 0) * np.maximum(yb - ya, 0)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Area under the precision-recall curve (VOC 11-point interpolation)."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        mask = recalls >= thr
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return ap


def compute_map(
    all_preds: list[dict],
    all_gts:   list[dict],
    iou_thresholds: list[float] = None,
) -> dict[str, float]:
    """
    Compute mAP@0.5 and mAP@0.5:0.95 for drone class.

    all_preds: list of dicts per image
        {'boxes': (N,4) x1y1x2y2, 'scores': (N,), 'labels': (N,)}
    all_gts: list of dicts per image
        {'boxes': (M,4) x1y1x2y2}
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 1.00, 0.05).tolist()

    aps: dict[float, float] = {}
    for iou_thr in iou_thresholds:
        # Collect all detections + gt matches
        tp_list, fp_list, scores_list = [], [], []
        n_gt = sum(len(g["boxes"]) for g in all_gts)

        for preds, gts in zip(all_preds, all_gts):
            boxes_p  = np.array(preds["boxes"],  dtype=np.float32)
            scores_p = np.array(preds["scores"], dtype=np.float32)
            boxes_g  = np.array(gts["boxes"],    dtype=np.float32)

            if len(boxes_p) == 0:
                continue

            # Sort by confidence
            order = np.argsort(-scores_p)
            boxes_p, scores_p = boxes_p[order], scores_p[order]
            scores_list.append(scores_p)

            if len(boxes_g) == 0:
                tp_list.append(np.zeros(len(boxes_p)))
                fp_list.append(np.ones(len(boxes_p)))
                continue

            iou_mat = box_iou(boxes_p, boxes_g)
            matched_gt = set()
            tp = np.zeros(len(boxes_p))
            fp = np.zeros(len(boxes_p))
            for i in range(len(boxes_p)):
                best = iou_mat[i].argmax()
                if iou_mat[i, best] >= iou_thr and best not in matched_gt:
                    tp[i] = 1
                    matched_gt.add(best)
                else:
                    fp[i] = 1
            tp_list.append(tp)
            fp_list.append(fp)

        if not scores_list:
            aps[iou_thr] = 0.0
            continue

        all_scores = np.concatenate(scores_list)
        all_tp     = np.concatenate(tp_list)
        all_fp     = np.concatenate(fp_list)

        # Sort globally by confidence
        order  = np.argsort(-all_scores)
        all_tp = all_tp[order]
        all_fp = all_fp[order]

        cum_tp = np.cumsum(all_tp)
        cum_fp = np.cumsum(all_fp)
        recalls    = cum_tp / max(n_gt, 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)

        aps[iou_thr] = compute_ap(recalls, precisions)

    map50    = aps.get(0.50, 0.0)
    map50_95 = float(np.mean(list(aps.values())))
    return {"mAP@0.5": map50, "mAP@0.5:0.95": map50_95}


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_gt_labels(labels_root: Path) -> dict[str, np.ndarray]:
    """
    Load all FRED test labels.

    Returns dict: {stem → (M, 4) boxes in x1y1x2y2, absolute pixels}

    Labels are stored per-sequence:
      labels_root/sequences/seq_*/labels/*.txt  (YOLO format)
    alongside rgb frames in labels_root/sequences/seq_*/rgb/*.jpg

    We look up image sizes to de-normalise boxes.
    """
    gt: dict[str, np.ndarray] = {}

    seq_root = labels_root / "sequences"
    if not seq_root.exists():
        log.warning("No sequences dir found at %s", seq_root)
        return gt

    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        lbl_dir = seq_dir / "labels"
        rgb_dir = seq_dir / "rgb"

        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            stem = lbl_path.stem
            # Find corresponding image to get H, W
            img_path = None
            for ext in (".jpg", ".png"):
                p = rgb_dir / (stem + ext)
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            rows = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    _, cx, cy, w, h = [float(v) for v in parts[:5]]
                    x1 = (cx - w / 2) * W
                    y1 = (cy - h / 2) * H
                    x2 = (cx + w / 2) * W
                    y2 = (cy + h / 2) * H
                    rows.append([x1, y1, x2, y2])

            gt[f"{seq_dir.name}/{stem}"] = (
                np.array(rows, dtype=np.float32) if rows
                else np.zeros((0, 4), dtype=np.float32)
            )

    return gt


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------

def run_detection_pipeline(
    frames_root: Path,
    labels_root: Path,
    conf: float = 0.25,
    iou: float  = 0.45,
    drone_coco_ids: Optional[list[int]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, float]:
    """
    Run YOLOv8n (COCO-pretrained) on all PNG frames in frames_root,
    match against FRED ground-truth labels, return mAP scores.

    frames_root layout must mirror reconstruct.py output:
        frames_root/seq_001/000001.png …

    labels_root is the FRED test split root (contains sequences/).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics  # required for detection")

    if drone_coco_ids is None:
        drone_coco_ids = DEFAULT_DRONE_COCO_IDS

    log.info("Loading YOLOv8n (pretrained) …")
    yolo = YOLO("yolov8n.pt")   # auto-downloads on first call

    gt_map = load_gt_labels(labels_root)
    if not gt_map:
        log.warning("No ground-truth labels found in %s — mAP will be 0", labels_root)

    all_preds, all_gts = [], []
    missing_gt = 0

    for seq_dir in sorted(frames_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        for png_path in sorted(seq_dir.glob("*.png")):
            stem  = png_path.stem
            key   = f"{seq_dir.name}/{stem}"

            # Grayscale → BGR for YOLO
            img_gray = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            H, W = img_bgr.shape[:2]

            result = yolo(img_bgr, conf=conf, iou=iou, verbose=False)[0]
            boxes  = result.boxes

            dets_boxes  = []
            dets_scores = []
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                if cls not in drone_coco_ids:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                dets_boxes.append([x1, y1, x2, y2])
                dets_scores.append(float(boxes.conf[i].item()))

            all_preds.append({
                "boxes":  dets_boxes,
                "scores": dets_scores,
                "labels": [0] * len(dets_boxes),
            })

            if key in gt_map:
                all_gts.append({"boxes": gt_map[key].tolist()})
            else:
                all_gts.append({"boxes": []})
                missing_gt += 1

    if missing_gt:
        log.warning("GT labels missing for %d frames (treated as negatives)", missing_gt)

    if not all_preds:
        log.warning("No frames processed — check frames_root: %s", frames_root)
        return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0}

    metrics = compute_map(all_preds, all_gts)
    log.info("Detection result: mAP@0.5=%.4f  mAP@0.5:0.95=%.4f",
             metrics["mAP@0.5"], metrics["mAP@0.5:0.95"])
    return metrics
