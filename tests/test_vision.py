import base64

import cv2
import numpy as np
import pytest
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResponseMetadata
from viam.services.mlmodel import Metadata, MLModel, TensorInfo
from viam.utils import dict_to_struct

from anomaly_vision import AnomalyVision
from decode import squeeze_map

W, H = 64, 48  # original image size
S = 32  # model input size


def make_config(**attrs):
    return ComponentConfig(name="anomaly", api="rdk:service:vision", model="viam-labs:anomalib:vision", attributes=dict_to_struct(attrs))


def make_image() -> ViamImage:
    _, buf = cv2.imencode(".jpg", np.full((H, W, 3), 128, np.uint8))
    return ViamImage(buf.tobytes(), CameraMimeType.JPEG)


def hot_map() -> np.ndarray:
    """Two hot regions in model space: cols 2..5 and 20..29."""
    m = np.zeros((1, 1, S, S), np.float32)
    m[0, 0, 2:6, 2:6] = 0.9
    m[0, 0, 20:28, 20:30] = 0.8
    return m


class FakeMLModel(MLModel):
    def __init__(self, score=0.7, outputs=("pred_score", "anomaly_map")):  # matches the stripped export
        super().__init__("ml")
        self.score = score
        self.output_names = list(outputs)
        self.inputs = []

    async def infer(self, input_tensors, *, extra=None, timeout=None):
        self.inputs.append(input_tensors)
        out = {"pred_score": np.array([self.score], np.float32), "anomaly_map": hot_map()}
        return {k: v for k, v in out.items() if k in self.output_names}

    async def metadata(self, *, extra=None, timeout=None):
        return Metadata(
            name="fake",
            type="fake",
            input_info=[TensorInfo(name="input", data_type="float32", shape=[1, 3, S, S])],
            output_info=[TensorInfo(name=n, data_type="float32") for n in self.output_names],
        )


class FakeCamera(Camera):
    def __init__(self, name: str, image: ViamImage):
        super().__init__(name)
        self.image = image

    async def get_images(self, *, filter_source_names=None, extra=None, timeout=None, **kwargs):
        return [NamedImage("color", self.image.data, self.image.mime_type)], ResponseMetadata()

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError

    async def get_properties(self, *, timeout=None, **kwargs):
        raise NotImplementedError


def make_service(ml, cam=None, **attrs):
    deps = {MLModel.get_resource_name(ml.name): ml}
    if cam is not None:
        deps[Camera.get_resource_name(cam.name)] = cam
    return AnomalyVision.new(make_config(mlmodel_name=ml.name, **attrs), deps)


def test_validate_config():
    assert AnomalyVision.validate_config(make_config(mlmodel_name="ml", camera_name="cam")) == (["ml"], ["cam"])
    assert AnomalyVision.validate_config(make_config(mlmodel_name="ml")) == (["ml"], [])
    with pytest.raises(ValueError, match="mlmodel_name"):
        AnomalyVision.validate_config(make_config())
    with pytest.raises(ValueError, match="threshold"):
        AnomalyVision.validate_config(make_config(mlmodel_name="ml", threshold=1.5))


@pytest.mark.parametrize("shape", [(1, 1, S, S), (1, S, S), (S, S)])
def test_squeeze_map(shape):
    assert squeeze_map(np.zeros(shape)).shape == (S, S)


async def test_sends_resized_float_input():
    ml = FakeMLModel()
    await make_service(ml).get_classifications(make_image(), 1)
    t = ml.inputs[0]["input"]
    assert t.shape == (1, 3, S, S) and t.dtype == np.float32
    assert 0.0 <= t.min() and t.max() <= 1.0


@pytest.mark.parametrize("score,label,conf", [(0.7, "anomalous", 0.7), (0.2, "anomalous", 0.2), (0.1, "normal", 0.9)])
async def test_classification(score, label, conf):
    [c] = await make_service(FakeMLModel(score=score)).get_classifications(make_image(), 1)
    assert c.class_name == label
    assert c.confidence == pytest.approx(conf)


async def test_threshold_attribute():
    [c] = await make_service(FakeMLModel(score=0.7), threshold=0.8).get_classifications(make_image(), 1)
    assert c.class_name == "normal"


async def test_missing_outputs_lists_found_names():
    with pytest.raises(ValueError, match="output0"):
        await make_service(FakeMLModel(outputs=("output0",))).get_classifications(make_image(), 1)


async def test_from_camera_requires_a_camera():
    with pytest.raises(ValueError, match="camera"):
        await make_service(FakeMLModel()).get_classifications_from_camera("", 1)


async def test_properties():
    p = await make_service(FakeMLModel()).get_properties()
    assert p.classifications_supported and p.detections_supported and not p.object_point_clouds_supported


async def test_detections_scaled_to_original_image():
    dets = await make_service(FakeMLModel(score=0.9)).get_detections(make_image())
    assert len(dets) == 2
    small, big = sorted(dets, key=lambda d: d.x_min)
    # model cols 2..5 and 20..29 of 32 -> image cols ~4..12 and ~40..60 of 64 (bilinear, threshold 0.2)
    assert 2 <= small.x_min <= 6 and 10 <= small.x_max <= 14
    assert 38 <= big.x_min <= 42 and 58 <= big.x_max <= 62
    for d in dets:
        assert d.class_name == "anomaly"
        assert 0 <= d.x_min_normalized <= d.x_max_normalized <= 1
        assert 0 < d.confidence <= 1


async def test_no_detections_when_image_is_normal():
    assert await make_service(FakeMLModel(score=0.1)).get_detections(make_image()) == []


async def test_capture_all_from_default_camera():
    cam = FakeCamera("cam", make_image())
    svc = make_service(FakeMLModel(score=0.9), cam, camera_name="cam")
    res = await svc.capture_all_from_camera("", return_image=True, return_classifications=True, return_detections=True)
    assert res.image is not None
    assert res.classifications[0].class_name == "anomalous"
    assert len(res.detections) == 2


async def test_heatmap_command():
    cam = FakeCamera("cam", make_image())
    out = await make_service(FakeMLModel(score=0.9), cam, camera_name="cam").do_command({"command": "heatmap"})
    assert out["label"] == "anomalous" and out["score"] == pytest.approx(0.9)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(out["image"]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (H, W, 3)
