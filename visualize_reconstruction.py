#!/usr/bin/env python3
"""
Convert reconstructed PNG frames to MP4 video.

Usage:
    python visualize_reconstruction.py [--input data/reconstructed] [--output data/reconstructed.mp4] [--fps 30]
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def make_video(input_dir: Path, output_path: Path, fps: int) -> None:
    import subprocess
    frames = sorted(input_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {input_dir}")

    print(f"{len(frames)} frames  {fps} fps → {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # H.264 via ffmpeg — required for browser playback
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(input_dir / "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",   # browser-compatible pixel format
        "-movflags", "+faststart",  # stream-friendly: moov atom at front
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    print(f"Saved: {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/reconstructed", help="Directory of PNG frames")
    p.add_argument("--output", default="data/reconstructed.mp4", help="Output video path")
    p.add_argument("--fps",    type=int, default=30, help="Frames per second (default: 30)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_video(Path(args.input), Path(args.output), args.fps)
