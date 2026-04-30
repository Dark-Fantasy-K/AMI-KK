# Hybrid Vision — AMI 2026

**Drone detection by fusing RGB frames and event-camera data.**

> AMI 2026 Course Project · FRED Dataset · PyTorch · Docker

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Quick Start](#quick-start)
4. [Phase 1 — Hybrid Detection Model](#phase-1--hybrid-detection-model)
5. [Phase 2 — Event Reconstruction](#phase-2--event-reconstruction)
6. [Phase 3 — Web Frontend](#phase-3--web-frontend)
7. [Project Structure](#project-structure)
8. [Requirements](#requirements)

---

## Overview

Traditional RGB cameras fail in challenging conditions — motion blur, high dynamic range, low light.
Event cameras address these limitations by asynchronously reporting per-pixel brightness changes at microsecond resolution.
This project explores *hybrid perception*: combining the complementary strengths of both sensor types to detect drones more robustly.

Three self-contained phases build toward a fully dockerised demo:

| Phase | Goal | Method |
|-------|------|--------|
| 1 | Detect drones from fused RGB + event data | Two-stream CNN with feature fusion |
| 2 | Reconstruct intensity frames from raw events | E2VID · FireNet (pretrained, no retraining) |
| 3 | Interactive web demo | FastAPI backend · HTML/JS frontend |

---

## Dataset

**FRED — Florence RGB-Event Drone Dataset**  
Homepage: <https://miccunifi.github.io/FRED/>

| Property | Value |
|----------|-------|
| Sensor | RGB camera + DAVIS event camera (synchronised) |
| Content | Outdoor drone flights, varied backgrounds |
| Format | RGB frames (JPEG) · Events (x, y, t, p) · YOLO labels |
| Split | Official train / test |

Expected on-disk layout after download:

```
data/FRED/
├── train/
│   └── sequences/
│       ├── seq_001/
│       │   ├── rgb/          # 000001.jpg …
│       │   ├── events/       # events.npy  (x y t p)
│       │   └── labels/       # 000001.txt  (YOLO format)
│       └── …
└── test/
    └── sequences/
        └── …
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/Dark-Fantasy-K/AMI-KK.git
cd AMI-KK

# Place FRED data under data/FRED/  (see Dataset section above)

# Launch everything (API + frontend)
docker-compose up --build
```

Open <http://localhost> in your browser.

---

## Phase 1 — Hybrid Detection Model

### Architecture

```
RGB frame  (3 ch)  ──►  RGB Encoder  ──►  RGB Features   ─┐
                                                            ├─► Fusion ─► FPN ─► Detection Head
Event frame (1 ch) ──►  Event Encoder ─►  Event Features  ─┘
```

Two fusion modes are supported (set via `configs/config.yaml`):

| Mode | Description |
|------|-------------|
| `early` | Stack RGB + event frame into a 4-channel tensor before the backbone |
| `late` | Separate ResNet18 encoders; fuse feature maps at each FPN level |

The detection head is a 3-scale YOLO-style head (anchors at /8, /16, /32 strides).
Output: bounding boxes + objectness + class (`drone`).

### Training

```bash
# Download FRED, set paths in configs/config.yaml, then:
python models/detection/hybrid/train.py \
    --config configs/config.yaml

# Checkpoints saved to path set in config (default: /data/checkpoints/)
```

### Key Files

| File | Role |
|------|------|
| [models/detection/hybrid/model.py](models/detection/hybrid/model.py) | `HybridDetector`: backbone, FPN, YOLO head |
| [preprocessing/dataset.py](preprocessing/dataset.py) | `FREDDataset`: loads RGB + events → 4-ch tensor |
| [preprocessing/event_processor.py](preprocessing/event_processor.py) | Events → voxel grid / event frame |
| [configs/config.yaml](configs/config.yaml) | All hyperparameters |

---

## Phase 2 — Event Reconstruction

### Motivation

Phase 2 answers a different question: *what if we skip fusion entirely and instead reconstruct a conventional intensity image from events alone, then feed it to a standard RGB detector?*

This establishes a baseline that isolates the contribution of the event camera without any specialised fusion architecture.
Both reconstruction networks are used in **inference-only** mode — no retraining is needed.

---

### Method 1 — E2VID

**Paper**: Rebecq et al., *"Events-to-Video: Bringing Modern Computer Vision to Event Cameras"*, CVPR 2019 / TPAMI 2021  
**Repo**: <https://github.com/uzh-rpg/rpg_e2vid>

#### Architecture

```
Voxel Grid         Recurrent UNet
(5 × H × W)  ──►  head (Conv)
                    │
                   enc[0]  ConvLSTM  stride=2  →  H/2 ,  64 ch
                   enc[1]  ConvLSTM  stride=2  →  H/4 , 128 ch
                   enc[2]  ConvLSTM  stride=2  →  H/8 , 256 ch
                    │
                  ResBlock × 2  (bottleneck)
                    │
                   dec[0]  ×2 upsample  →  H/4 , 128 ch  ⊕ skip enc[1]
                   dec[1]  ×2 upsample  →  H/2 ,  64 ch  ⊕ skip enc[0]
                   dec[2]  ×2 upsample  →  H   ,  32 ch  ⊕ skip head
                    │
                   pred (1×1 Conv) + Sigmoid
                    │
               Reconstructed Frame (H × W, float32)
```

**Key design choices**:
- Input is a **voxel grid** (5 time bins, bilinear interpolation across bins) rather than a simple event frame — preserves temporal structure within each inter-frame interval.
- **Recurrent hidden states** (ConvLSTM) carry context across consecutive frames — crucial for smooth reconstruction of slow-moving content.
- **Sum skip connections** from each encoder level prevent information loss from strided convolutions.
- Our implementation ([models/reconstruction/e2vid/model.py](models/reconstruction/e2vid/model.py)) mirrors the official rpg_e2vid parameter names so the **official pretrained checkpoint loads directly** without any key remapping.

#### Usage

```bash
# 1. Download pretrained weights
bash models/reconstruction/e2vid/download_weights.sh /data/weights

# 2. Reconstruct all FRED test sequences
python models/reconstruction/e2vid/reconstruct.py \
    --fred_root /data/FRED \
    --weights   /data/weights/e2vid.pth \
    --output    /data/reconstructed/e2vid \
    --num_bins  5 \
    --device    cuda
```

Output: `/data/reconstructed/e2vid/seq_XXX/000001.png …`

---

### Method 2 — FireNet

**Paper**: Scheerlinck et al., *"Fast Image Reconstruction with an Event Camera"*, WACV 2020  
**Repo**: <https://github.com/cedric-scheerlinck/rpg_e2vid> (firenet branch)

#### Architecture

```
Voxel Grid         FireNet
(5 × H × W)  ──►  head  (Conv → BN → ReLU)
                    │
                   G1  ConvLSTM  (16 ch, full resolution)
                   G2  ConvLSTM
                   G3  ConvLSTM
                    │
                   pred  (1×1 Conv) + Sigmoid
                    │
               Reconstructed Frame (H × W, float32)
```

**Comparison with E2VID**:

| Aspect | E2VID | FireNet |
|--------|-------|---------|
| Architecture | UNet (encoder-decoder) | Linear chain |
| Parameters | ~5 M | ~0.1 M |
| Downsampling | 3× stride-2 conv | None |
| Spatial context | Multi-scale via FPN | Single scale |
| Speed | Slower | **Faster** |
| Quality (typical) | Higher | Good for fast motion |

FireNet trades reconstruction quality for speed — useful when latency matters more than fine detail.

**Checkpoint compatibility**: `load_firenet()` handles two checkpoint variants automatically:
- Separate `ih` / `hh` convolutions (standard FireNet format)
- Combined `Gates` convolution (some variants) — split on load

#### Usage

```bash
# 1. Download pretrained weights
bash models/reconstruction/method2/download_weights.sh /data/weights

# 2. Reconstruct
python models/reconstruction/method2/reconstruct.py \
    --fred_root /data/FRED \
    --weights   /data/weights/firenet.pth \
    --output    /data/reconstructed/firenet \
    --device    cuda
```

---

### Evaluation

After reconstruction, run the three-way comparison:

```bash
python evaluation/compare.py \
    --fred_root      /data/FRED \
    --e2vid_frames   /data/reconstructed/e2vid \
    --firenet_frames /data/reconstructed/firenet \
    --hybrid_checkpoint /data/checkpoints/best.pt   # optional
```

Sample output:

```
==============================================================
  FRED Detection Comparison
==============================================================
Method                               mAP@0.5     mAP@0.5:0.95    Time (s)
-----------------------------------  ----------  --------------  ----------
E2VID → YOLOv8 (COCO)               0.XXXX      0.XXXX          XX.X
FireNet → YOLOv8 (COCO)             0.XXXX      0.XXXX          XX.X
Hybrid (RGB + Event, Phase 1)        0.XXXX      0.XXXX          XX.X
==============================================================
```

Detection on reconstructed frames uses **YOLOv8n** (pretrained on COCO, auto-downloaded) as a frozen baseline detector.
The mAP difference between the E2VID and FireNet rows therefore reflects purely reconstruction quality — no detector fine-tuning is involved.

### Phase 2 File Map

```
models/reconstruction/
├── utils.py                          # shared: event loading, voxel grid, frame saving
├── e2vid/
│   ├── model.py                      # E2VID network (checkpoint-compatible)
│   ├── reconstruct.py                # FRED test reconstruction script
│   └── download_weights.sh           # gdown from Google Drive
└── method2/                          # FireNet
    ├── model.py                      # FireNet network
    ├── reconstruct.py                # same interface as E2VID
    └── download_weights.sh

evaluation/
├── detect_on_frames.py               # YOLOv8 inference + mAP computation
└── compare.py                        # prints the three-way table
```

---

## Phase 3 — Web Frontend

Phase 3 wraps the full pipeline in a browser-accessible demo:
- Upload an RGB frame and a raw event file (`.npy` / `.h5`)
- Choose detection mode: *Hybrid* (Phase 1 model) or *Reconstruct + Detect* (Phase 2)
- Results are displayed as bounding-box overlays in real time

### Services

| Service | Image | Port | Role |
|---------|-------|------|------|
| `api` | `webapp/backend/Dockerfile` | 8000 | FastAPI inference server |
| `frontend` | `webapp/frontend/Dockerfile` | 80 | nginx static HTML/JS |

```bash
docker-compose up --build
# Frontend → http://localhost
# API docs  → http://localhost:8000/docs
```

---

## Project Structure

```
AMI-KK/
├── configs/
│   └── config.yaml                   # all hyperparameters
├── data/                             # mount FRED here
├── evaluation/
│   ├── compare.py                    # three-way mAP table
│   ├── detect_on_frames.py           # YOLOv8 + mAP on reconstructed frames
│   └── metrics.py                   # mAP helpers
├── models/
│   ├── detection/hybrid/
│   │   ├── model.py                  # HybridDetector
│   │   ├── train.py                  # training loop
│   │   └── predict.py               # inference
│   └── reconstruction/
│       ├── utils.py                  # shared event/frame utilities
│       ├── e2vid/
│       │   ├── model.py              # E2VID (Phase 2, Method 1)
│       │   ├── reconstruct.py
│       │   └── download_weights.sh
│       └── method2/
│           ├── model.py              # FireNet (Phase 2, Method 2)
│           ├── reconstruct.py
│           └── download_weights.sh
├── preprocessing/
│   ├── dataset.py                    # FREDDataset (PyTorch)
│   └── event_processor.py           # events → voxel grid / event frame
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── download_data.sh
├── webapp/
│   ├── backend/
│   │   ├── app.py                    # FastAPI
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── frontend/
│       ├── index.html
│       ├── nginx.conf
│       └── Dockerfile
└── docker-compose.yml
```

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ (optional, CPU fallback available) |
| Docker + Compose | 24+ |

**Key Python packages**:

```
torch torchvision
ultralytics          # YOLOv8 (Phase 2 detection baseline)
fastapi uvicorn python-multipart
opencv-python-headless
numpy scipy h5py pyyaml
gdown                # checkpoint download
```

---

## License

MIT — see `LICENSE`.

---

*AMI 2026 · Hybrid Vision Project*
