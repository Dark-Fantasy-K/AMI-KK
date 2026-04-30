#!/usr/bin/env bash
# =============================================================================
# FRED Dataset Download & Setup Script
# =============================================================================
#
# Source : https://huggingface.co/datasets/GabrieleMagrini/FRED
# Mirror : https://drive.google.com/drive/folders/1pISIErXOx76xmCqkwhS3-azWOMlTKZMp
#
# Pipeline:
#   1. List available sequences on HuggingFace (no download)
#   2. Select the first N train + M test sequences
#   3. Download only the selected sequences (not the full dataset)
#   4. Extract, organise into train/ and test/ splits
#   5. Convert annotations → YOLO format
#   6. Validate structure
#
# Split rule (official FRED):
#   Sequences 000–099  →  train
#   Sequences 100+     →  test
#
# Annotation native format (coordinates_rgb.txt):
#   time: x1, y1, x2, y2, id, class   (absolute pixel coordinates)
#
# YOLO output format (one .txt per RGB frame):
#   0 cx_norm cy_norm w_norm h_norm
#
# Usage:
#   bash scripts/download_data.sh [OPTIONS]
#
# Options:
#   -o, --output  DIR   Output directory            (default: ./data/FRED)
#   -t, --train   N     Number of train sequences   (default: 0 = all)
#   -e, --test    N     Number of test sequences    (default: 0 = all)
#   -l, --list          List available sequences and exit (no download)
#   -h, --help          Show this help message
#
# Examples:
#   bash scripts/download_data.sh --train 5 --test 2
#   bash scripts/download_data.sh --output /data/FRED --train 20 --test 5
#   bash scripts/download_data.sh --list
#   bash scripts/download_data.sh                      # download everything
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()    { echo -e "\n${BOLD}${CYAN}▶ $*${NC}"; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
FRED_ROOT="$(pwd)/data/FRED"
MAX_TRAIN=0          # 0 = download all available
MAX_TEST=0           # 0 = download all available
LIST_ONLY=false
HF_REPO="GabrieleMagrini/FRED"
GDRIVE_FOLDER="1pISIErXOx76xmCqkwhS3-azWOMlTKZMp"
TRAIN_THRESHOLD=100  # seq numbers 0–99 → train, 100+ → test

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    grep '^#' "$0" | grep -A100 'Usage:' | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            FRED_ROOT="$2"; shift 2 ;;
        -t|--train)
            [[ "$2" =~ ^[0-9]+$ ]] || error "--train requires a non-negative integer"
            MAX_TRAIN="$2"; shift 2 ;;
        -e|--test)
            [[ "$2" =~ ^[0-9]+$ ]] || error "--test requires a non-negative integer"
            MAX_TEST="$2"; shift 2 ;;
        -l|--list)
            LIST_ONLY=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        -*)
            error "Unknown option: $1  (run with --help for usage)" ;;
        *)
            # Legacy positional: first arg = output dir
            FRED_ROOT="$1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}FRED Dataset Downloader${NC}"
echo    "  Output : $FRED_ROOT"
if [[ "$MAX_TRAIN" -eq 0 ]]; then
    echo "  Train  : all sequences"
else
    echo "  Train  : first $MAX_TRAIN sequences"
fi
if [[ "$MAX_TEST" -eq 0 ]]; then
    echo "  Test   : all sequences"
else
    echo "  Test   : first $MAX_TEST sequences"
fi
echo

# ---------------------------------------------------------------------------
# Step 0 — Ensure huggingface_hub is available
# ---------------------------------------------------------------------------
step "Checking dependencies"

ensure_hf_hub() {
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        warn "huggingface_hub not found — installing …"
        pip install -q "huggingface_hub>=0.23"
    fi
    info "huggingface_hub OK"
}

ensure_hf_hub

# ---------------------------------------------------------------------------
# Step 1 — List available sequences (no download)
# ---------------------------------------------------------------------------
step "Listing available sequences on HuggingFace"

SEQ_LIST_JSON=$(python3 - <<PYEOF
import json, re, sys
from huggingface_hub import list_repo_files

try:
    all_files = list(list_repo_files("$HF_REPO", repo_type="dataset"))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(0)

train, test = [], []
for f in all_files:
    m = re.search(r'(\d+)\.zip', f)
    if not m:
        continue
    n = int(m.group(1))
    entry = {"num": n, "file": f}
    (train if n < $TRAIN_THRESHOLD else test).append(entry)

train.sort(key=lambda x: x["num"])
test.sort(key=lambda x: x["num"])
print(json.dumps({"train": train, "test": test}))
PYEOF
)

# Check for error
if echo "$SEQ_LIST_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if 'error' not in d else 1)" 2>/dev/null; then
    :
else
    ERR=$(echo "$SEQ_LIST_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin).get('error','unknown'))")
    error "Failed to list HuggingFace files: $ERR"
fi

# Print available counts
python3 - "$SEQ_LIST_JSON" "$MAX_TRAIN" "$MAX_TEST" <<'PYEOF'
import json, sys

data      = json.loads(sys.argv[1])
max_train = int(sys.argv[2])
max_test  = int(sys.argv[3])
train     = data["train"]
test      = data["test"]

sel_train = train if max_train == 0 else train[:max_train]
sel_test  = test  if max_test  == 0 else test[:max_test]

print(f"  Available  — train: {len(train):3d}  test: {len(test):3d}")
print(f"  Will fetch — train: {len(sel_train):3d}  test: {len(sel_test):3d}")
if sel_train:
    nums = [e['num'] for e in sel_train]
    print(f"  Train seqs : {nums[0]:03d} … {nums[-1]:03d}")
if sel_test:
    nums = [e['num'] for e in sel_test]
    print(f"  Test seqs  : {nums[0]:03d} … {nums[-1]:03d}")
PYEOF

# --list mode: just print the full table and exit
if $LIST_ONLY; then
    echo
    python3 - "$SEQ_LIST_JSON" <<'PYEOF'
import json, sys
data = json.loads(sys.argv[1])
print(f"\n{'Split':<6} {'Seq':>4}  {'File'}")
print("-" * 60)
for entry in data["train"]:
    print(f"  train  {entry['num']:03d}  {entry['file']}")
for entry in data["test"]:
    print(f"  test   {entry['num']:03d}  {entry['file']}")
PYEOF
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2 — Download selected sequences
# ---------------------------------------------------------------------------
step "Downloading selected sequences"

mkdir -p "$FRED_ROOT"
TMP_DIR="$FRED_ROOT/.download_tmp"
mkdir -p "$TMP_DIR/zips"

python3 - "$SEQ_LIST_JSON" "$MAX_TRAIN" "$MAX_TEST" "$TMP_DIR/zips" "$HF_REPO" <<'PYEOF'
import json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

data      = json.loads(sys.argv[1])
max_train = int(sys.argv[2])
max_test  = int(sys.argv[3])
dest      = Path(sys.argv[4])
repo      = sys.argv[5]

sel_train = data["train"] if max_train == 0 else data["train"][:max_train]
sel_test  = data["test"]  if max_test  == 0 else data["test"][:max_test]
selected  = sel_train + sel_test

if not selected:
    print("No sequences match the requested counts — nothing to download.")
    sys.exit(0)

for i, entry in enumerate(selected, 1):
    fname = entry["file"]
    local = dest / Path(fname).name
    if local.exists():
        print(f"  [{i}/{len(selected)}] Already cached: {local.name}")
        continue
    print(f"  [{i}/{len(selected)}] Downloading {fname} …", flush=True)
    hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=fname,
        local_dir=str(dest),
    )

print(f"\nAll {len(selected)} sequences downloaded to {dest}")
PYEOF

# ---------------------------------------------------------------------------
# Step 3 — Extract archives
# ---------------------------------------------------------------------------
step "Extracting sequence archives"

mkdir -p "$TMP_DIR/extracted"

find "$TMP_DIR/zips" -name "*.zip" | sort | while read -r zipfile; do
    seq_name=$(basename "$zipfile" .zip)
    out_dir="$TMP_DIR/extracted/$seq_name"
    if [ -d "$out_dir" ]; then
        info "  Already extracted: $seq_name"
        continue
    fi
    info "  Extracting $seq_name …"
    unzip -q "$zipfile" -d "$out_dir" || warn "  Failed to extract $zipfile — skipping."
done

# ---------------------------------------------------------------------------
# Step 4 — Organise into train / test splits
# ---------------------------------------------------------------------------
step "Organising into train / test splits"

mkdir -p "$FRED_ROOT/train/sequences"
mkdir -p "$FRED_ROOT/test/sequences"

ORGANISED=0

find "$TMP_DIR/extracted" -mindepth 1 -maxdepth 1 -type d | sort | while read -r seq_dir; do
    seq_name=$(basename "$seq_dir")
    seq_num=$(echo "$seq_name" | grep -oE '[0-9]+' | head -1)

    if [ -z "$seq_num" ]; then
        warn "Cannot determine index for '$seq_name' — skipping."
        continue
    fi

    if [ "$seq_num" -lt "$TRAIN_THRESHOLD" ]; then
        SPLIT="train"
    else
        SPLIT="test"
    fi

    DEST="$FRED_ROOT/$SPLIT/sequences/seq_$(printf '%03d' "$seq_num")"
    if [ -d "$DEST" ]; then
        info "  Already organised: seq_$(printf '%03d' "$seq_num") ($SPLIT)"
        continue
    fi
    mkdir -p "$DEST"

    # RGB frames
    for rgb_src in "$seq_dir"/rgb "$seq_dir"/frames "$seq_dir"/images; do
        if [ -d "$rgb_src" ]; then
            cp -r "$rgb_src" "$DEST/rgb"
            break
        fi
    done
    find "$DEST/rgb" -name "*.jpeg" \
        -exec bash -c 'mv "$1" "${1%.jpeg}.jpg"' _ {} \; 2>/dev/null || true

    # Event data (HDF5)
    mkdir -p "$DEST/events"
    for evt_src in \
        "$seq_dir"/*.hdf5 "$seq_dir"/*.h5 \
        "$seq_dir"/events/*.hdf5 "$seq_dir"/events/*.h5; do
        if [ -f "$evt_src" ]; then
            cp "$evt_src" "$DEST/events/events.h5"
            break
        fi
    done
    if [ -d "$seq_dir/event_frames" ]; then
        cp -r "$seq_dir/event_frames/." "$DEST/events/"
    fi

    # Raw annotations (kept for reference; YOLO labels created in step 5)
    mkdir -p "$DEST/labels"
    for ann_src in "$seq_dir"/coordinates_rgb.txt "$seq_dir"/coordinates.txt; do
        if [ -f "$ann_src" ]; then
            cp "$ann_src" "$DEST/labels/_raw_annotations.txt"
            break
        fi
    done

    info "  seq_$(printf '%03d' "$seq_num") → $SPLIT"
    ORGANISED=$((ORGANISED + 1))
done

# ---------------------------------------------------------------------------
# Step 5 — Convert annotations → YOLO format
# ---------------------------------------------------------------------------
step "Converting annotations to YOLO format"

python3 - "$FRED_ROOT" <<'PYEOF'
"""
Convert FRED native annotations to per-frame YOLO label files.

Native format (coordinates_rgb.txt):
    time: x1, y1, x2, y2, id, class   (absolute pixel coords)

YOLO output (one .txt per RGB frame, zero-indexed class):
    0  cx_norm  cy_norm  w_norm  h_norm

Matching: each RGB frame is linearly mapped to the annotation timeline
and matched to the nearest annotated timestamp.
"""
import sys, re, cv2
from pathlib import Path
from collections import defaultdict

fred_root     = Path(sys.argv[1])
total_frames  = 0
total_boxes   = 0
skipped_seqs  = 0

for split in ("train", "test"):
    seq_root = fred_root / split / "sequences"
    if not seq_root.exists():
        continue

    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue

        ann_file = seq_dir / "labels" / "_raw_annotations.txt"
        rgb_dir  = seq_dir / "rgb"
        lbl_dir  = seq_dir / "labels"

        if not ann_file.exists() or not rgb_dir.exists():
            skipped_seqs += 1
            continue

        # Parse native annotations
        annots = defaultdict(list)
        with open(ann_file) as f:
            for line in f:
                nums = re.findall(r"[\d.]+", line)
                if len(nums) < 5:
                    continue
                t, x1, y1, x2, y2 = (float(v) for v in nums[:5])
                annots[t].append((x1, y1, x2, y2))

        if not annots:
            skipped_seqs += 1
            continue

        sorted_times = sorted(annots.keys())
        t_min, t_max = sorted_times[0], sorted_times[-1]

        rgb_frames = sorted(
            list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png"))
        )
        if not rgb_frames:
            continue

        img0 = cv2.imread(str(rgb_frames[0]))
        if img0 is None:
            continue
        H, W = img0.shape[:2]

        n = len(rgb_frames)
        for i, rgb_path in enumerate(rgb_frames):
            # Skip if YOLO label already exists
            lbl_path = lbl_dir / (rgb_path.stem + ".txt")
            if lbl_path.exists():
                total_frames += 1
                total_boxes  += sum(1 for _ in open(lbl_path))
                continue

            # Map frame index linearly to annotation timeline
            frame_t  = t_min + (t_max - t_min) * i / max(n - 1, 1)
            nearest  = min(sorted_times, key=lambda t: abs(t - frame_t))
            boxes    = annots[nearest]

            with open(lbl_path, "w") as out:
                for (x1, y1, x2, y2) in boxes:
                    x1, x2 = max(0.0, min(float(x1), W)), max(0.0, min(float(x2), W))
                    y1, y2 = max(0.0, min(float(y1), H)), max(0.0, min(float(y2), H))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cx = ((x1 + x2) / 2) / W
                    cy = ((y1 + y2) / 2) / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H
                    out.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    total_boxes  += 1
            total_frames += 1

print(f"  {total_boxes} boxes written for {total_frames} frames "
      f"({skipped_seqs} sequences skipped — no annotations).")
PYEOF

# ---------------------------------------------------------------------------
# Step 6 — Validate
# ---------------------------------------------------------------------------
step "Validating dataset structure"

python3 - "$FRED_ROOT" <<'PYEOF'
import sys
from pathlib import Path

root   = Path(sys.argv[1])
ok     = 0
issues = 0

for split in ("train", "test"):
    seqs = sorted((root / split / "sequences").glob("seq_*"))
    if not seqs:
        continue
    print(f"\n  {split}/ ({len(seqs)} sequences)")
    print(f"  {'Seq':<10} {'RGB':>5}  {'Labels':>6}  {'events.h5'}")
    print(f"  {'-'*10} {'-'*5}  {'-'*6}  {'-'*9}")
    for seq in seqs:
        rgb_n  = len(list((seq/'rgb').glob('*.jpg'))) + len(list((seq/'rgb').glob('*.png')))
        lbl_n  = len([f for f in (seq/'labels').glob('*.txt')
                      if not f.name.startswith('_')])
        has_h5 = (seq/'events'/'events.h5').exists()
        status = '✓' if (rgb_n > 0 and lbl_n > 0) else '!'
        if status == '!':
            issues += 1
        else:
            ok += 1
        marker = '' if status == '✓' else '  ← missing data'
        print(f"  [{status}] {seq.name:<8}  {rgb_n:5d}  {lbl_n:6d}  {'yes' if has_h5 else 'no'}{marker}")

print(f"\n  Result: {ok} OK  |  {issues} with issues")
if issues:
    print("  Sequences marked '!' may have missing annotations or event files.")
PYEOF

# ---------------------------------------------------------------------------
# Step 7 — Cleanup
# ---------------------------------------------------------------------------
step "Cleaning up"
rm -rf "$TMP_DIR"
info "Temporary files removed."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
echo -e "${BOLD}${GREEN}✓ FRED dataset ready at: $FRED_ROOT${NC}"
echo
echo "Next steps:"
echo "  1. Update configs/config.yaml → dataset.root: $FRED_ROOT"
echo "  2. Train the hybrid model:"
echo "       python models/detection/hybrid/train.py --config configs/config.yaml"
echo "  3. Download reconstruction weights and reconstruct:"
echo "       bash models/reconstruction/e2vid/download_weights.sh /data/weights"
echo "       python models/reconstruction/e2vid/reconstruct.py \\"
echo "           --fred_root $FRED_ROOT --weights /data/weights/e2vid.pth \\"
echo "           --output /data/reconstructed/e2vid"
echo "  4. Compare all pipelines:"
echo "       python evaluation/compare.py --fred_root $FRED_ROOT \\"
echo "           --e2vid_frames /data/reconstructed/e2vid \\"
echo "           --firenet_frames /data/reconstructed/firenet"
echo
