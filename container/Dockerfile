# viamlabs/anomalib-train: environment for the anomalib Viam training script (script itself lives in the registry).
FROM --platform=linux/amd64 python:3.12-slim

# libGL/glib for opencv inside anomalib.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# PyPI torch wheels for linux/amd64 ship CUDA. openvino pinned to match erh:openvino's runtime.
RUN pip install --no-cache-dir "anomalib[openvino]==2.7.0" "openvino==2026.4.0" torch torchvision
