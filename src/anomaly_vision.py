"""viam-labs:anomalib:vision - a vision service that decodes Anomalib outputs from any mlmodel service."""

import asyncio
import base64
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import Self
from viam.components.camera import Camera
from viam.errors import NotSupportedError
from viam.logging import getLogger
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, Detection3D, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model
from viam.services.mlmodel import MLModel
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes, struct_to_dict

from decode import classify, decode_image, find_boxes, heatmap_jpeg, squeeze_map, to_input

LOGGER = getLogger(__name__)
REQUIRED_OUTPUTS = ("pred_score", "anomaly_map")


class AnomalyConfig:
    def __init__(self, attrs: Mapping[str, Any]) -> None:
        name = attrs.get("mlmodel_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("mlmodel_name is required and must name an mlmodel service")
        self.mlmodel_name: str = name.strip()

        cam = attrs.get("camera_name")
        if cam is not None and (not isinstance(cam, str) or not cam.strip()):
            raise ValueError("camera_name must be a non-empty string when set")
        self.camera_name: Optional[str] = cam.strip() if cam else None

        t = attrs.get("threshold", 0.5)
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0.0 <= t <= 1.0:
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = float(t)


class AnomalyVision(Vision, EasyResource):
    MODEL: ClassVar[Model] = "viam-labs:anomalib:vision"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.cfg: Optional[AnomalyConfig] = None
        self.mlmodel: Optional[MLModel] = None
        self.cameras: Dict[str, Camera] = {}
        self._info: Optional[Tuple[str, int, int]] = None  # input name, H, W
        self._info_lock = asyncio.Lock()

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = cls(config.name)
        self.reconfigure(config, dependencies)
        return self

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        cfg = AnomalyConfig(struct_to_dict(config.attributes))
        return [cfg.mlmodel_name], [cfg.camera_name] if cfg.camera_name else []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        cfg = AnomalyConfig(struct_to_dict(config.attributes))
        ml = dependencies.get(MLModel.get_resource_name(cfg.mlmodel_name))
        if ml is None:
            raise ValueError(f"mlmodel '{cfg.mlmodel_name}' was not found in dependencies")
        self.mlmodel = ml  # type: ignore[assignment]
        self.cameras = {rn.name: res for rn, res in dependencies.items() if rn.subtype == "camera"}  # type: ignore[misc]
        self.cfg = cfg
        self._info = None

    # ---------------------------------------------------------------- inference
    async def _model_info(self) -> Tuple[str, int, int]:
        if self._info is not None:
            return self._info
        async with self._info_lock:
            if self._info is None:
                assert self.mlmodel is not None and self.cfg is not None
                md = await self.mlmodel.metadata()
                found = [o.name for o in md.output_info]
                missing = [n for n in REQUIRED_OUTPUTS if n not in found]
                if missing:
                    raise ValueError(
                        f"mlmodel '{self.cfg.mlmodel_name}' outputs {found} lack {missing}; "
                        "the model does not look like an Anomalib export"
                    )
                inp = md.input_info[0]
                shape = list(inp.shape)
                if len(shape) != 4 or shape[2] <= 0 or shape[3] <= 0:
                    raise ValueError(f"mlmodel input shape {shape} must be a fixed [1, 3, H, W]; export with input_size set")
                self._info = (inp.name, int(shape[2]), int(shape[3]))
                LOGGER.info("vision '%s': input '%s' %dx%d, outputs %s", self.name, inp.name, shape[2], shape[3], found)
        return self._info

    async def _analyze(self, image: ViamImage) -> Tuple[np.ndarray, float, np.ndarray]:
        assert self.mlmodel is not None
        name, h, w = await self._model_info()
        rgb = decode_image(image.data)
        out = await self.mlmodel.infer({name: to_input(rgb, h, w)})
        return rgb, float(np.asarray(out["pred_score"]).reshape(-1)[0]), squeeze_map(out["anomaly_map"])

    def _classifications(self, score: float) -> List[Classification]:
        assert self.cfg is not None
        label, conf = classify(score, self.cfg.threshold)
        return [Classification(class_name=label, confidence=conf)]

    def _detections(self, rgb: np.ndarray, score: float, amap: np.ndarray) -> List[Detection]:
        assert self.cfg is not None
        if score < self.cfg.threshold:
            return []  # image is normal: no boxes, so capture filters stay predictable
        h, w = rgb.shape[:2]
        return [
            Detection(
                x_min=x1, y_min=y1, x_max=x2, y_max=y2,
                x_min_normalized=x1 / w, y_min_normalized=y1 / h, x_max_normalized=x2 / w, y_max_normalized=y2 / h,
                confidence=conf, class_name="anomaly",
            )
            for x1, y1, x2, y2, conf in find_boxes(amap, self.cfg.threshold, w, h)
        ]

    async def _camera_image(self, camera_name: str) -> ViamImage:
        assert self.cfg is not None
        name = camera_name or self.cfg.camera_name
        if not name:
            raise ValueError(f"vision '{self.name}': no camera_name given and no default camera configured")
        cam = self.cameras.get(name)
        if cam is None:
            raise ValueError(f"vision '{self.name}': camera '{name}' is not a dependency; set camera_name to '{name}'")
        images, _ = await cam.get_images()
        if not images:
            raise RuntimeError(f"camera '{name}' returned no images")
        return ViamImage(images[0].data, images[0].mime_type)

    # --------------------------------------------------------------- vision API
    async def get_classifications(
        self, image: ViamImage, count: int, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None
    ) -> List[Classification]:
        _, score, _ = await self._analyze(image)
        return self._classifications(score)

    async def get_classifications_from_camera(
        self, camera_name: str, count: int, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None
    ) -> List[Classification]:
        return await self.get_classifications(await self._camera_image(camera_name), count)

    async def get_detections(
        self, image: ViamImage, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None
    ) -> List[Detection]:
        return self._detections(*await self._analyze(image))

    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None
    ) -> List[Detection]:
        return await self.get_detections(await self._camera_image(camera_name))

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        image = await self._camera_image(camera_name)
        if return_image:
            result.image = image
        if return_classifications or return_detections:
            rgb, score, amap = await self._analyze(image)
            if return_classifications:
                result.classifications = self._classifications(score)
            if return_detections:
                result.detections = self._detections(rgb, score, amap)
        return result

    async def get_object_point_clouds(
        self, camera_name: str, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        raise NotSupportedError(f"vision '{self.name}' (viam-labs:anomalib:vision) does not support object point clouds")

    async def get_detections_3d(
        self, camera_name: str, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None
    ) -> List[Detection3D]:
        # Abstract in viam-sdk >= 0.85 (erh:openvino pins 0.83, which predates it).
        raise NotSupportedError(f"vision '{self.name}' (viam-labs:anomalib:vision) does not support 3D detections")

    async def get_properties(
        self, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False,
            default_camera=(self.cfg.camera_name or "") if self.cfg else "",
        )

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        if command.get("command") != "heatmap":
            raise ValueError(f"unknown command {command.get('command')!r}; expected 'heatmap'")
        assert self.cfg is not None
        rgb, score, amap = await self._analyze(await self._camera_image(str(command.get("camera_name") or "")))
        label, _ = classify(score, self.cfg.threshold)
        return {"score": score, "label": label, "image": base64.b64encode(heatmap_jpeg(rgb, amap)).decode()}
