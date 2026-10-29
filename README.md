# viam-labs:anomalib

Visual anomaly detection on Viam using [Anomalib](https://github.com/open-edge-platform/anomalib), from data capture to deployment.
You show it good parts, train in the Viam cloud, and run on Intel CPU, GPU or NPU through OpenVINO.

This repo provides:

- **`viam-labs:anomalib:vision`** — a vision service. It turns the output of an Anomalib model, served by
  [`erh:openvino:mlmodel`](https://app.viam.com/module/erh/openvino), into classifications, detections and heatmaps.
- **`anomalib-patchcore`** — a custom training script. It trains PatchCore by default and exports an OpenVINO IR (`model.xml`/`model.bin`).
- **`hipsterbrown/anomalib-train:2.7.0-cu128`** — the training container (anomalib 2.7.0 + OpenVINO 2026.4).

## Quickstart

1. **Capture.** Add a camera and data capture to your machine, then point the camera at *good* parts.
2. **Tag.** In DATA, add at least 10 images of good parts to a dataset and tag them `normal`.
   Optional: tag 4 or more images of defective parts `anomalous`. Labeled defects give a more meaningful threshold and metrics.
   If you skip them, training evaluates against synthetic anomalies instead.
3. **Train.** In the hackathons org, the script and container are already registered:
   ```bash
   viam train submit custom from-registry --org-id=<your org> --dataset-id=<your dataset> \
     --model-name=my-anomaly --model-version=1 \
     --script-name=hackathons:anomalib-patchcore --version=0.1.1 \
     --container-id=9accd6e7-2992-47c9-9662-2ca5298fa1f1 \
     --args=model=patchcore,image_size=256
   ```
   `viam train logs --job-id=<id>` shows the image AUROC and F1 at the end.
4. **Deploy.** Add the `my-anomaly` model to your machine. If the `erh:openvino` and `viam-labs:anomalib` modules
   don't come with it, add these two services:
   ```json
   { "name": "ov", "api": "rdk:service:mlmodel", "model": "erh:openvino:mlmodel",
     "attributes": { "model_path": "${packages.ml_model.my-anomaly}/model.xml" } }
   ```
   ```json
   { "name": "anomaly", "api": "rdk:service:vision", "model": "viam-labs:anomalib:vision",
     "attributes": { "mlmodel_name": "ov", "camera_name": "cam" } }
   ```
   **Tip: run PatchCore on the GPU in f32.** Add `"device": "GPU", "inference_precision": "f32"` to the `ov` attributes.
   On a Core Ultra X7 358H, mean latency per 256×256 image was:

   | Device | Mean latency |
   |---|---|
   | Arc iGPU, f32 | 31 ms |
   | CPU | 77 ms |
   | NPU | 158 ms |

   The default `AUTO` picks the NPU, which is the slowest of the three for this model.
   Keep f32: with the GPU's default f16, `pred_score` is still correct but `anomaly_map` comes back all zeros, so no boxes and an empty heatmap.
   If OpenVINO doesn't list a `GPU` device, the host's Intel compute runtime is probably too old for the iGPU.
   See the [erh:viam-openvino README](https://github.com/erh/viam-openvino#host-prerequisites-for-gpu-and-npu).
5. **Use.**
   - `GetClassifications` returns `anomalous` (confidence = anomaly score) or `normal` (confidence = 1 − score).
   - `GetDetections` returns boxes around anomalous regions, or no boxes when the score is below `threshold`.
   - `DoCommand {"command": "heatmap"}` returns `score`, `label`, and `image`, which is the camera frame with a heatmap overlay as a base64 JPEG.
   - `CaptureAllFromCamera` returns the image, classifications and detections in a single call.

## Vision service attributes

| Attribute | Required | Default | |
|---|---|---|---|
| `mlmodel_name` | yes | | mlmodel service that serves the Anomalib export, e.g. `erh:openvino:mlmodel` |
| `camera_name` | no | | default camera for `*FromCamera` calls |
| `threshold` | no | `0.2` | anomaly score cutoff, 0–1. Lower it if defects are missed; raise it if real lighting causes false positives |

## Training args

| Arg | Default | |
|---|---|---|
| `model` | `patchcore` | any Anomalib image model name |
| `image_size` | `256` | square input size in pixels |

The export takes one RGB image, `input` `[1,3,S,S]` with float values from 0 to 1, and produces two outputs,
`pred_score` `[1]` and `anomaly_map` `[1,1,S,S]`.

No parts? The hackathons org has an `mvtec-bottle` dataset (`6ae0082a78f241c0f6d45593`) with 209 `normal` and 20
`anomalous` images from MVTec AD. It can be recreated in any org with `scripts/upload_mvtec.py`.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pytest
```

The vision module ships for linux/amd64 and windows/amd64 as a PyInstaller binary. Pushing a `v*` tag publishes it
(see `.github/workflows/deploy.yml`). On Windows, install a current Intel graphics driver to use the GPU or NPU.
The training container is built from `container/Dockerfile`.
