#!/usr/bin/env python3
import json as _json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reconstruct" / "firenet"))

from utils.loading_utils import load_model as _load_ckpt
from utils.inference_utils import CropParameters, events_to_voxel_grid_pytorch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS = {
    "e2vid":   ROOT / "reconstruct/e2vid/pretrained/E2VID_lightweight.pth.tar",
    "firenet": ROOT / "reconstruct/firenet/pretrained/firenet_1000.pth.tar",
}
JOBS_DIR = ROOT / ".jobs"
JOBS_DIR.mkdir(exist_ok=True)

# Clean up frame dirs from previous server sessions (videos are kept)
for _d in JOBS_DIR.iterdir():
    if _d.is_dir() and (_d / "frames").exists():
        shutil.rmtree(_d / "frames", ignore_errors=True)

# Set USE_REFERENCE_SCRIPT=1 to delegate reconstruction to run_reconstruction_npy.py
# (uses ImageReconstructor: unsharp mask, bilateral filter, exact reference timing)
USE_REFERENCE_SCRIPT = os.environ.get("USE_REFERENCE_SCRIPT", "0") == "1"

# ---------------------------------------------------------------------------
# Model cache  (lazy-load per model type, GPU by default)
# ---------------------------------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_models: dict[str, torch.nn.Module] = {}
_model_lock = threading.Lock()

torch.set_num_threads(os.cpu_count() or 1)
if _device.type == "cuda":
    torch.backends.cudnn.benchmark = True


def get_model(model_type: str) -> torch.nn.Module:
    if model_type not in _models:
        model = _load_ckpt(str(WEIGHTS[model_type]))
        _models[model_type] = model.to(_device).eval()
    return _models[model_type]


# ---------------------------------------------------------------------------
# Reconstruction loop
# ---------------------------------------------------------------------------

@torch.inference_mode()
def reconstruct(model, data: np.ndarray, height: int, width: int,
                out_dir: Path, events_per_frame: int = 0,
                progress_cb=None) -> tuple[int, float]:
    """Returns (n_frames, inference_ms) where inference_ms excludes PNG writes."""
    num_bins = model.num_bins
    crop     = CropParameters(width, height, model.num_encoders)
    if events_per_frame <= 0:
        events_per_frame = int(width * height * 0.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    states       = None
    n            = len(data)
    n_frames     = (n + events_per_frame - 1) // events_per_frame
    hdr_bounds   = deque(maxlen=5)
    inference_ms = 0.0

    write_thread: threading.Thread | None = None

    def _write(path: str, img: np.ndarray) -> None:
        cv2.imwrite(path, img)

    with torch.autocast(device_type=_device.type, enabled=(_device.type == "cuda")):
        for i in range(n_frames):
            chunk = data[i * events_per_frame : (i + 1) * events_per_frame]

            t0 = time.perf_counter()
            x = events_to_voxel_grid_pytorch(chunk, num_bins, width, height, _device)
            x = crop.pad(x.unsqueeze(0))
            pred, states = model(x, states)
            pred = pred[:, :, crop.iy0:crop.iy1, crop.ix0:crop.ix1].float()

            lo_hi = pred.aminmax()          # GPU→CPU sync
            Imin  = float(lo_hi.min.clamp(0.0, 0.45))
            Imax  = float(lo_hi.max.clamp(0.55, 1.0))
            hdr_bounds.append((Imin, Imax))
            lo = float(np.median([b[0] for b in hdr_bounds]))
            hi = float(np.median([b[1] for b in hdr_bounds]))

            # Convert to uint8 on GPU (4× smaller GPU→CPU transfer, matches IntensityRescaler)
            img = ((pred[0, 0] - lo) / (hi - lo)).clamp(0, 1).mul(255).byte()
            img = img.cpu().numpy()
            inference_ms += (time.perf_counter() - t0) * 1000

            if write_thread is not None:
                write_thread.join()
            write_thread = threading.Thread(
                target=_write, args=(str(out_dir / f"{i:06d}.png"), img), daemon=True
            )
            write_thread.start()

            if progress_cb:
                progress_cb(i + 1, n_frames)

    if write_thread is not None:
        write_thread.join()

    return n_frames, inference_ms


# ---------------------------------------------------------------------------
# Event frame visualization (no model — positive=white, negative=blue)
# ---------------------------------------------------------------------------

def make_event_frames(data: np.ndarray, height: int, width: int,
                      out_dir: Path, events_per_frame: int = 0,
                      progress_cb=None) -> int:
    if events_per_frame <= 0:
        events_per_frame = int(width * height * 0.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    n        = len(data)
    n_frames = (n + events_per_frame - 1) // events_per_frame

    write_thread: threading.Thread | None = None

    def _write(path: str, img: np.ndarray) -> None:
        cv2.imwrite(path, img)

    for i in range(n_frames):
        # Compute xs/ys/ps per-chunk — avoids allocating 3 × N × 4 B for the full dataset
        chunk = data[i * events_per_frame : (i + 1) * events_per_frame]
        xs  = np.clip(chunk[:, 1].astype(np.int32), 0, width  - 1)
        ys  = np.clip(chunk[:, 2].astype(np.int32), 0, height - 1)
        pos = chunk[:, 3] == 1

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[ys[ pos], xs[ pos]] = (255, 255, 255)  # positive → white
        frame[ys[~pos], xs[~pos]] = (200,  80,  30)  # negative → blue (BGR)

        if write_thread is not None:
            write_thread.join()
        write_thread = threading.Thread(
            target=_write, args=(str(out_dir / f"{i:06d}.png"), frame), daemon=True
        )
        write_thread.start()

        if progress_cb:
            progress_cb(i + 1, n_frames)

    if write_thread is not None:
        write_thread.join()

    return n_frames


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------

def make_video(frames_dir: Path, out_path: Path, fps: int = 30) -> None:
    subprocess.run(
        ["ffmpeg", "-y",
         "-framerate", str(fps),
         "-i", str(frames_dir / "%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(out_path)],
        check=True, capture_output=True,
    )


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}

_ADAPTER = ROOT / "reconstruct" / "firenet" / "run_reconstruction_npy.py"


def run_job_subprocess(job_id: str, npy_path: Path, model_type: str,
                       events_per_frame: int, delete_npy: bool = True) -> None:
    """Delegate reconstruction to the reference script via subprocess.
    Uses ImageReconstructor (with unsharp mask, bilateral filter, auto HDR).
    """
    job        = jobs[job_id]
    frames_dir = JOBS_DIR / job_id / "frames"
    video_path = JOBS_DIR / job_id / "output.mp4"
    frames_dir.mkdir(parents=True, exist_ok=True)
    job["frames_dir"] = str(frames_dir)  # expose path for streaming endpoint

    try:
        cmd = [
            sys.executable, str(_ADAPTER),
            "--path_to_model", str(WEIGHTS[model_type]),
            "--npy_file",      str(npy_path),
            "--output_folder", str(frames_dir.parent),
            "--dataset_name",  "frames",
            "--auto_hdr",
            "--use_gpu",
        ]
        if events_per_frame > 0:
            cmd += ["--window_size", str(events_per_frame)]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                cwd=str(ROOT / "reconstruct" / "firenet"), text=True)

        job.update(status="reconstructing", done=0, total=0, message="Reconstructing…")
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = _json.loads(line)
                job.update(done=d["done"], total=d["total"],
                           message=f"Reconstructing {d['done']}/{d['total']} frames…")
            except Exception:
                pass

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Reconstruction subprocess exited with code {proc.returncode}")

        job.update(status="encoding", message="Encoding video…")
        t_enc = time.perf_counter()
        make_video(frames_dir, video_path, fps=30)
        encode_ms = (time.perf_counter() - t_enc) * 1000

        job.update(status="done", message="Done!", video=str(video_path),
                   report={"n_frames": job.get("total", 0), "encode_ms": round(encode_ms),
                           "inference_ms": 0, "gpu_peak_mb": 0, "fps": 0})

    except Exception as e:
        job.update(status="error", message=str(e))
    finally:
        if delete_npy:
            npy_path.unlink(missing_ok=True)


def run_job(job_id: str, npy_path: Path, model_type: str,
            events_per_frame: int, delete_npy: bool = True) -> None:
    job        = jobs[job_id]
    frames_dir = JOBS_DIR / job_id / "frames"
    video_path = JOBS_DIR / job_id / "output.mp4"
    job["frames_dir"] = str(frames_dir)  # expose path for streaming endpoint

    try:
        job.update(status="loading", message="Loading events…")
        data   = np.load(npy_path, mmap_mode='c')  # demand-paged; doesn't pin whole array
        height = int(data[:, 2].max()) + 1
        width  = int(data[:, 1].max()) + 1

        model    = get_model(model_type)
        n_frames = (len(data) + events_per_frame - 1) // events_per_frame if events_per_frame > 0 \
                   else (len(data) + int(width * height * 0.35) - 1) // int(width * height * 0.35)
        job.update(status="reconstructing", total=n_frames, done=0,
                   message=f"Reconstructing 0/{n_frames} frames…")

        if _device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(_device)

        with _model_lock:
            n_frames, inference_ms = reconstruct(
                model, data, height, width,
                out_dir=frames_dir,
                events_per_frame=events_per_frame,
                progress_cb=lambda done, total: job.update(
                    done=done, message=f"Reconstructing {done}/{total} frames…"
                ),
            )

        gpu_peak_mb = (torch.cuda.max_memory_allocated(_device) / 1e6
                       if _device.type == "cuda" else 0.0)

        job.update(status="encoding", message="Encoding video…")
        t_enc = time.perf_counter()
        make_video(frames_dir, video_path, fps=30)
        encode_ms = (time.perf_counter() - t_enc) * 1000

        report = {
            "n_frames":     n_frames,
            "inference_ms": round(inference_ms),
            "encode_ms":    round(encode_ms),
            "gpu_peak_mb":  round(gpu_peak_mb),
            "fps":          round(n_frames / max(inference_ms / 1000, 1e-6), 1),
        }
        job.update(status="done", message="Done!", video=str(video_path), report=report)

    except Exception as e:
        job.update(status="error", message=str(e))
    finally:
        if delete_npy:
            npy_path.unlink(missing_ok=True)


def run_event_preview_job(job_id: str, npy_path: Path, events_per_frame: int,
                          delete_npy: bool = True) -> None:
    job        = jobs[job_id]
    frames_dir = JOBS_DIR / job_id / "frames"
    video_path = JOBS_DIR / job_id / "output.mp4"
    job["frames_dir"] = str(frames_dir)  # expose path for streaming endpoint

    try:
        job.update(status="loading", message="Loading events…")
        data   = np.load(npy_path, mmap_mode='c')  # demand-paged
        height = int(data[:, 2].max()) + 1
        width  = int(data[:, 1].max()) + 1

        epf      = events_per_frame if events_per_frame > 0 else int(width * height * 0.35)
        n_frames = (len(data) + epf - 1) // epf
        job.update(status="rendering", total=n_frames, done=0,
                   message=f"Rendering 0/{n_frames} frames…")

        make_event_frames(
            data, height, width,
            out_dir=frames_dir,
            events_per_frame=events_per_frame,
            progress_cb=lambda done, total: job.update(
                done=done, message=f"Rendering {done}/{total} frames…"
            ),
        )

        job.update(status="encoding", message="Encoding video…")
        make_video(frames_dir, video_path, fps=30)
        job.update(status="done", message="Done!", video=str(video_path))

    except Exception as e:
        job.update(status="error", message=str(e))
    finally:
        if delete_npy:
            npy_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# App & routes
# ---------------------------------------------------------------------------
app = FastAPI(title="E2VID Reconstruction API")


def _save_upload(upload: UploadFile, dest: Path) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


@app.post("/api/reconstruct")
def api_reconstruct(
    file: UploadFile = File(...),
    model: str = Form("e2vid"),
    events_per_frame: int = Form(0),
):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="File must be .npy")
    if model not in WEIGHTS:
        raise HTTPException(status_code=400, detail=f"model must be one of: {list(WEIGHTS)}")

    job_id   = str(uuid.uuid4())
    npy_path = JOBS_DIR / f"{job_id}.npy"
    (JOBS_DIR / job_id).mkdir(parents=True, exist_ok=True)
    _save_upload(file, npy_path)

    _recon_fn = run_job_subprocess if USE_REFERENCE_SCRIPT else run_job
    jobs[job_id] = {"status": "queued", "message": "Queued",
                    "done": 0, "total": 0, "model": model}
    threading.Thread(
        target=_recon_fn,
        args=(job_id, npy_path, model, events_per_frame),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.post("/api/event_preview")
def api_event_preview(
    file: UploadFile = File(...),
    events_per_frame: int = Form(0),
):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="File must be .npy")

    job_id   = str(uuid.uuid4())
    npy_path = JOBS_DIR / f"{job_id}.npy"
    (JOBS_DIR / job_id).mkdir(parents=True, exist_ok=True)
    _save_upload(file, npy_path)

    jobs[job_id] = {"status": "queued", "message": "Queued", "done": 0, "total": 0}
    threading.Thread(
        target=run_event_preview_job,
        args=(job_id, npy_path, events_per_frame),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.post("/api/full_job")
def api_full_job(
    file: UploadFile = File(...),
    model: str = Form("e2vid"),
    events_per_frame: int = Form(0),
):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="File must be .npy")
    if model not in WEIGHTS:
        raise HTTPException(status_code=400, detail=f"model must be one of: {list(WEIGHTS)}")

    recon_id = str(uuid.uuid4())
    prev_id  = str(uuid.uuid4())
    (JOBS_DIR / recon_id).mkdir(parents=True, exist_ok=True)
    (JOBS_DIR / prev_id).mkdir(parents=True, exist_ok=True)

    # Save once; both jobs mmap the same file.
    # Preview finishes first (no model) and skips deletion.
    # Reconstruction deletes the file when it's done.
    shared_npy = JOBS_DIR / f"{recon_id}.npy"
    _save_upload(file, shared_npy)

    jobs[recon_id] = {"status": "queued", "message": "Queued",
                      "done": 0, "total": 0, "model": model}
    jobs[prev_id]  = {"status": "queued", "message": "Queued", "done": 0, "total": 0}

    _recon_fn = run_job_subprocess if USE_REFERENCE_SCRIPT else run_job
    threading.Thread(target=run_event_preview_job,
                     args=(prev_id, shared_npy, events_per_frame),
                     kwargs={"delete_npy": False},
                     daemon=True).start()
    threading.Thread(target=_recon_fn,
                     args=(recon_id, shared_npy, model, events_per_frame),
                     daemon=True).start()

    return {"reconstruct_job_id": recon_id, "preview_job_id": prev_id}


@app.get("/api/status/{job_id}")
def api_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/video/{job_id}")
def api_video(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Video not ready")
    return FileResponse(job["video"], media_type="video/mp4")


@app.get("/api/frame/{job_id}/{index}")
def api_frame(job_id: str, index: int):
    """Serve an individual PNG frame for streaming preview during inference."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="")
    frames_dir = job.get("frames_dir")
    if not frames_dir:
        raise HTTPException(status_code=404, detail="")
    path = Path(frames_dir) / f"{index:06d}.png"
    # Retry briefly in case the write thread hasn't flushed yet
    for _ in range(20):
        if path.exists():
            return FileResponse(str(path), media_type="image/png")
        time.sleep(0.01)
    raise HTTPException(status_code=404, detail="")


# Static files must be mounted last — catches all routes not matched above
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print(f"Device : {_device}")
    for name, path in WEIGHTS.items():
        print(f"{name:8}: {path}")
    uvicorn.run(app, host="0.0.0.0", port=5000)
