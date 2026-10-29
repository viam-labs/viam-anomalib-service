# hipsterbrown/anomalib-train: environment for the anomalib Viam training script (script itself lives in the registry).
FROM --platform=linux/amd64 python:3.12-slim

# libGL/glib for opencv inside anomalib.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# torch from the CUDA 12.8 index: the CUDA line Viam's training containers run on (a cu130 build needs an R580+ driver).
# openvino pinned to match erh:openvino's runtime.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
      "torch==2.11.0+cu128" torchvision "anomalib[openvino]==2.7.0" "openvino==2026.4.0" \
 && python -c "import torch, anomalib; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'anomalib', anomalib.__version__)"

# python:3.12-slim isn't an NVIDIA CUDA image, so declare what nvidia-container-runtime should expose.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Google's GPU nodes mount the driver (libcuda, nvidia-smi) here; torch's wheels bring the rest of CUDA.
ENV PATH=/usr/local/nvidia/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib
