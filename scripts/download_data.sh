#!/usr/bin/env bash
set -euo pipefail

OUT="./data/FRED"
TRAIN=0
TEST=0
while [[ $# -gt 0 ]]; do
  case "$1" in -o|--out) OUT="$2"; shift 2;; -t|--train) TRAIN="$2"; shift 2;; -e|--test) TEST="$2"; shift 2;; *) echo "Usage: $0 [-o DIR] [-t N] [-e N]"; exit 1;; esac
done

mkdir -p "$OUT/train" "$OUT/test"

for i in $(seq 0 $((TRAIN - 1))); do
  [ -d "$OUT/train/$i" ] && echo "skip train/$i" && continue
  wget -q --show-progress -O "/tmp/$i.zip" "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/$i.zip"
  unzip -qo "/tmp/$i.zip" -d "$OUT/train/" && rm "/tmp/$i.zip"
  echo "-> train/$i"
done

for i in $(seq 110 $((109 + TEST))); do
  [ -d "$OUT/test/$i" ] && echo "skip test/$i" && continue
  wget -q --show-progress -O "/tmp/$i.zip" "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/test/$i.zip"
  unzip -qo "/tmp/$i.zip" -d "$OUT/test/" && rm "/tmp/$i.zip"
  echo "-> test/$i"
done

echo "Done."