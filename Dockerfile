FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib2.0-0 satisfy opencv-python's dynamic library deps even
# though the dashboard never opens a GUI window (cv2.imshow isn't used
# here); curl is for the healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch/torchvision first (pinned to the versions
# verified against anomalib==2.6.0 locally) — without this, pip pulls
# the default GPU wheel with several GB of CUDA runtime libraries
# (nvidia-cublas, nvidia-cudnn, etc.) that this container never uses,
# since it only ever runs inference on CPU.
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
