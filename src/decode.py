"""Pure helpers for Anomalib model I/O: image decode, input tensor, map squeeze, classification, boxes, heatmap."""

from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int, float]  # x_min, y_min, x_max, y_max, confidence


def decode_image(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode camera image; expected JPEG or PNG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_input(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """RGB HxWx3 uint8 -> float32 [1, 3, h, w] in [0, 1]. The export does its own resize/normalize internally."""
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray((resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])


def squeeze_map(arr) -> np.ndarray:
    m = np.asarray(arr, dtype=np.float32)
    while m.ndim > 2 and m.shape[0] == 1:
        m = m[0]
    if m.ndim != 2:
        raise ValueError(f"anomaly_map must squeeze to HxW, got shape {list(np.shape(arr))}")
    return m


def classify(score: float, threshold: float) -> Tuple[str, float]:
    if score >= threshold:
        return "anomalous", score
    return "normal", 1.0 - score


def find_boxes(amap: np.ndarray, threshold: float, orig_w: int, orig_h: int) -> List[Box]:
    """One box per connected region of the map at or above threshold, in original image pixels."""
    m = cv2.resize(amap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m >= threshold).astype(np.uint8), connectivity=8)
    boxes: List[Box] = []
    for i in range(1, n):  # 0 is background
        x, y, w, h = (int(v) for v in stats[i, :4])
        boxes.append((x, y, x + w, y + h, min(float(m[labels == i].max()), 1.0)))
    return boxes


def heatmap_jpeg(rgb: np.ndarray, amap: np.ndarray) -> bytes:
    h, w = rgb.shape[:2]
    m = cv2.resize(np.clip(amap, 0.0, 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.5, color, 0.5, 0)
    _, buf = cv2.imencode(".jpg", blended)
    return buf.tobytes()
