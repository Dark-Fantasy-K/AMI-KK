FROM python:3.12-slim

WORKDIR /app

# System libs required by OpenCV + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.1 — wheels bundle the required CUDA runtime libs,
# so python:slim base is sufficient (no nvidia/cuda base image needed).
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies (torch already satisfied, pip skips it)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
