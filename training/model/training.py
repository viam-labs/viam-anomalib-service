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
    # Single-pass models (PatchCore, PaDiM, DFKDE, DFM) force max_epochs=1 themselves; this only bounds the iterative ones.
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--train_batch_size", type=int, default=32)  # EfficientAd requires 1
    args, unknown = p.parse_known_args()
    if unknown:
        LOGGER.warning("ignoring unknown args: %s", unknown)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    import anomalib
    import openvino as ov
    import torch
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.metrics import AUROC, Evaluator, F1Score
    from anomalib.models import get_model

    out = Path(args.model_output_directory)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="anomalib-"))
    counts = prepare(args.dataset_file, work / "data")
    size = (args.image_size, args.image_size)

    # Anomalib v2: image size goes through the model's pre-processor, not the datamodule.
    model_cls = type(get_model(args.model))  # ponytail: builds the model once just to find its class
    # Image metrics only: Viam datasets have no pixel masks, and some models (FastFlow) default to strict pixel metrics.
    evaluator = Evaluator(
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
        ]
    )
    model = get_model(args.model, pre_processor=model_cls.configure_pre_processor(image_size=size), evaluator=evaluator)

    has_anomalous = counts[ANOMALOUS] > 0
    datamodule = Folder(
        name="viam",
        root=work / "data",
        normal_dir=NORMAL,
        abnormal_dir=ANOMALOUS if has_anomalous else None,
        test_split_mode="from_dir" if has_anomalous else "synthetic",
        train_batch_size=args.train_batch_size,
    )

    # One device: Viam's GPU nodes have two, and Lightning would otherwise launch DDP, running this whole script per rank.
    engine = Engine(default_root_dir=work / "runs", max_epochs=args.max_epochs, devices=1)
    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    metrics = results[0] if results else {}

    # Convert directly at batch 1: engine.export traces with batch 2 and PatchCore's graph specializes to it.
    # Returning only the two float outputs also drops the bool ones, which break viam-sdk's mlmodel serialization.
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            r = self.m(x)  # pre_processor -> model -> post_processor
            return r.pred_score, r.anomaly_map

    model = model.cpu().eval()
    s = args.image_size
    ir = ov.convert_model(Wrapper(model), example_input=torch.zeros(1, 3, s, s), input=[1, 3, s, s])
    ir.input(0).tensor.set_names({"input"})
    ir.output(0).tensor.set_names({"pred_score"})
    ir.output(1).tensor.set_names({"anomaly_map"})
    ov.save_model(ir, out / "model.xml", compress_to_fp16=False)
    metadata = {
        "model": args.model,
        "image_size": args.image_size,
        "max_epochs": args.max_epochs,
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
