#!/usr/bin/env python3
import os
import sys
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.reconstruction.e2vid.model import load_e2vid
from models.reconstruction.e2vid.reconstruct import reconstruct
from models.reconstruction.utils import events_to_voxel
from visualize_reconstruction import make_video

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS = ROOT / "models/reconstruction/e2vid/pretrained_weights/E2VID_lightweight.pth.tar"
JOBS_DIR = Path("/tmp/e2vid_jobs")
JOBS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

# ---------------------------------------------------------------------------
# Model (loaded once, shared across requests with a lock)
# ---------------------------------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        _model = load_e2vid(WEIGHTS, _device)
        torch.set_num_threads(os.cpu_count() or 1)
    return _model


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}


def run_job(job_id: str, npy_path: Path, events_per_frame: int, num_bins: int):
    job = jobs[job_id]
    frames_dir = JOBS_DIR / job_id / "frames"
    video_path = JOBS_DIR / job_id / "output.mp4"

    try:
        job.update(status="loading", message="Loading events…")
        data = np.load(npy_path)
        events = {
            "t": data[:, 0].astype(np.float64),
            "x": data[:, 1].astype(np.int32),
            "y": data[:, 2].astype(np.int32),
            "p": ((data[:, 3] + 1) // 2).astype(np.int8),
        }
        height = int(events["y"].max()) + 1
        width  = int(events["x"].max()) + 1

        n = len(events["t"])
        n_frames = (n + events_per_frame - 1) // events_per_frame
        job.update(status="reconstructing", total=n_frames, done=0,
                   message=f"Reconstructing 0/{n_frames} frames…")

        def progress(done, total):
            job.update(done=done, message=f"Reconstructing {done}/{total} frames…")

        with _model_lock:
            model = get_model()
            reconstruct(
                model, events, height, width,
                out_dir=frames_dir,
                num_bins=num_bins,
                events_per_frame=events_per_frame,
                device=_device,
                progress_cb=progress,
            )

        job.update(status="encoding", message="Encoding video…")
        make_video(frames_dir, video_path, fps=30)
        shutil.rmtree(frames_dir, ignore_errors=True)

        job.update(status="done", message="Done!", video=str(video_path))

    except Exception as e:
        job.update(status="error", message=str(e))
    finally:
        npy_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.post("/api/reconstruct")
def api_reconstruct():
    if "file" not in request.files:
        return jsonify(error="No file provided"), 400

    f = request.files["file"]
    if not f.filename.endswith(".npy"):
        return jsonify(error="File must be .npy"), 400

    events_per_frame = int(request.form.get("events_per_frame", 200000))
    num_bins         = int(request.form.get("num_bins", 5))

    job_id   = str(uuid.uuid4())
    npy_path = JOBS_DIR / f"{job_id}.npy"
    (JOBS_DIR / job_id).mkdir(parents=True, exist_ok=True)
    f.save(str(npy_path))

    jobs[job_id] = {"status": "queued", "message": "Queued", "done": 0, "total": 0}
    threading.Thread(
        target=run_job,
        args=(job_id, npy_path, events_per_frame, num_bins),
        daemon=True,
    ).start()

    return jsonify(job_id=job_id)


@app.get("/api/status/<job_id>")
def api_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job not found"), 404
    return jsonify(job)


@app.get("/api/video/<job_id>")
def api_video(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify(error="Video not ready"), 404
    return send_file(job["video"], mimetype="video/mp4", as_attachment=False)


if __name__ == "__main__":
    print(f"Device: {_device}")
    print(f"Weights: {WEIGHTS}")
    app.run(host="0.0.0.0", port=5000, debug=False)
