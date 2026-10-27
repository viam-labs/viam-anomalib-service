import asyncio
import sys

from viam.module.module import Module

from anomaly_vision import AnomalyVision  # noqa: F401  (importing registers the model)


def _selftest() -> int:
    """Smoke test for packaged binaries: exercises the cv2/numpy native code paths without a robot."""
    import numpy as np

    from decode import find_boxes, heatmap_jpeg, to_input

    rgb = np.zeros((48, 64, 3), np.uint8)
    amap = np.zeros((32, 32), np.float32)
    amap[4:8, 4:8] = 1.0
    assert to_input(rgb, 32, 32).shape == (1, 3, 32, 32)
    assert len(find_boxes(amap, 0.5, 64, 48)) == 1
    assert heatmap_jpeg(rgb, amap)[:2] == b"\xff\xd8"
    print("selftest ok")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    asyncio.run(Module.run_from_registry())
