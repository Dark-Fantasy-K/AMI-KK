#!/usr/bin/env python3
"""
Adapter: run the reference ImageReconstructor on a .npy event file.
Accepts the same flags as run_reconstruction.py plus --npy_file.
Prints JSON progress lines to stdout: {"done": N, "total": M}
"""
import sys
import json
import argparse
import numpy as np
import torch

from utils.loading_utils import load_model, get_device
from utils.inference_utils import events_to_voxel_grid_pytorch
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--path_to_model', required=True)
    parser.add_argument('--npy_file', required=True)
    parser.add_argument('-N', '--window_size', default=None, type=int)
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float)
    parser.add_argument('--compute_voxel_grid_on_cpu', action='store_true')
    set_inference_options(parser)
    args = parser.parse_args()

    data   = np.load(args.npy_file, mmap_mode='c')
    height = int(data[:, 2].max()) + 1
    width  = int(data[:, 1].max()) + 1
    print(f'Sensor size: {width} x {height}', flush=True)

    model  = load_model(args.path_to_model)
    device = get_device(args.use_gpu)
    model  = model.to(device).eval()

    N = args.window_size or int(width * height * args.num_events_per_pixel)
    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    n      = len(data)
    total  = (n + N - 1) // N
    print(json.dumps({"done": 0, "total": total}), flush=True)

    for i in range(total):
        chunk = data[i * N : (i + 1) * N]
        last_ts = float(chunk[-1, 0])

        event_tensor = events_to_voxel_grid_pytorch(
            chunk, num_bins=model.num_bins,
            width=width, height=height, device=device
        )
        reconstructor.update_reconstruction(event_tensor, i * N + len(chunk), last_ts)
        print(json.dumps({"done": i + 1, "total": total}), flush=True)


if __name__ == '__main__':
    main()
