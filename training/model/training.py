"""Viam custom training script: train an Anomalib model on a Viam dataset and export OpenVINO IR."""

import argparse
import json
import logging
import tempfile
from pathlib import Path

from model.dataset import ANOMALOUS, NORMAL, prepare

LOGGER = logging.getLogger("anomalib-train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_file", required=True)
    p.add_argument("--model_output_directory", required=True)
    p.add_argument("--model", default="patchcore")
    p.add_argument("--image_size", type=int, default=256)
    args, unknown = p.parse_known_args()
    if unknown:
        LOGGER.warning("ignoring unknown args: %s", unknown)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    import anomalib
    import openvino as ov
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import get_model

    out = Path(args.model_output_directory)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="anomalib-"))
    counts = prepare(args.dataset_file, work / "data")
    size = (args.image_size, args.image_size)

    # Anomalib v2: image size goes through the model's pre-processor, not the datamodule.
    model_cls = type(get_model(args.model))  # ponytail: builds the model once just to find its class
    model = get_model(args.model, pre_processor=model_cls.configure_pre_processor(image_size=size))

    has_anomalous = counts[ANOMALOUS] > 0
    datamodule = Folder(
        name="viam",
        root=work / "data",
        normal_dir=NORMAL,
        abnormal_dir=ANOMALOUS if has_anomalous else None,
        test_split_mode="from_dir" if has_anomalous else "synthetic",
    )

    engine = Engine(default_root_dir=work / "runs")
    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    metrics = results[0] if results else {}

    # input_size is required, otherwise H/W export as dynamic and the vision service can't read them.
    engine.export(model=model, export_type=ExportType.OPENVINO, export_root=work / "export", input_size=size)
    xml = next((work / "export").rglob("model.xml"))

    # Keep only the float outputs the vision service reads. The bool outputs (pred_label, pred_mask) come back
    # from erh:openvino as np.bool_, which viam-sdk's mlmodel gRPC serialization rejects (KeyError: 'bool').
    full = ov.Core().read_model(xml)
    ir = ov.Model([full.output("pred_score"), full.output("anomaly_map")], full.get_parameters(), "anomalib")
    ov.save_model(ir, out / "model.xml", compress_to_fp16=False)
    metadata = {
        "model": args.model,
        "image_size": args.image_size,
        "input_name": ir.inputs[0].any_name,
        "output_names": [o.any_name for o in ir.outputs],
        "image_auroc": float(metrics.get("image_AUROC", float("nan"))),
        "image_f1": float(metrics.get("image_F1Score", float("nan"))),
        "n_normal": counts[NORMAL],
        "n_anomalous": counts[ANOMALOUS],
        "anomalib_version": anomalib.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    LOGGER.info("exported %s", json.dumps(metadata))


if __name__ == "__main__":
    main()
