"""Turn a Viam dataset JSONL into normal/ and anomalous/ image folders for anomalib's Folder datamodule."""

import json
import logging
import os
import shutil
from pathlib import Path

NORMAL, ANOMALOUS = "normal", "anomalous"
MIN_NORMAL = 10
MIN_ANOMALOUS = 4  # below this the val split taken from test can't calibrate a threshold

LOGGER = logging.getLogger(__name__)


def prepare(dataset_file, work_dir) -> dict:
    work = Path(work_dir)
    for label in (NORMAL, ANOMALOUS):
        (work / label).mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys((NORMAL, ANOMALOUS, "skipped_unlabeled", "skipped_unknown", "skipped_both"), 0)
    with open(dataset_file) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            tags = {a.get("annotation_label") for a in row.get("classification_annotations") or []}
            ours = tags & {NORMAL, ANOMALOUS}
            if len(ours) == 2:
                counts["skipped_both"] += 1
            elif not ours:
                counts["skipped_unknown" if tags else "skipped_unlabeled"] += 1
            else:
                label = ours.pop()
                src = Path(row["image_path"])
                os.symlink(src, work / label / f"{i:06d}{src.suffix}")
                counts[label] += 1
    LOGGER.info("dataset: %s", counts)
    if counts[NORMAL] < MIN_NORMAL:
        raise ValueError(f"need at least {MIN_NORMAL} images tagged '{NORMAL}', found {counts[NORMAL]}")
    if 0 < counts[ANOMALOUS] < MIN_ANOMALOUS:
        LOGGER.warning(
            "only %d '%s' images (< %d); ignoring them and using synthetic anomalies", counts[ANOMALOUS], ANOMALOUS, MIN_ANOMALOUS
        )
        shutil.rmtree(work / ANOMALOUS)
        (work / ANOMALOUS).mkdir()
        counts[ANOMALOUS] = 0
    return counts
