import json
import sys

import pytest

from model.dataset import prepare

# Training only runs in the linux container; creating symlinks needs admin rights on Windows CI runners.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="training data prep is linux-only")


def write_jsonl(tmp_path, rows):
    lines = []
    for i, labels in enumerate(rows):
        img = tmp_path / f"img{i}.jpg"
        img.write_bytes(b"x")
        lines.append(json.dumps({"image_path": str(img), "classification_annotations": [{"annotation_label": lb} for lb in labels]}))
    p = tmp_path / "dataset.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_splits_and_skips(tmp_path):
    rows = [["normal"]] * 10 + [["anomalous"]] * 4 + [[], ["scratch"], ["normal", "anomalous"], ["normal", "bottle"]]
    counts = prepare(write_jsonl(tmp_path, rows), tmp_path / "work")
    assert counts == {"normal": 11, "anomalous": 4, "skipped_unlabeled": 1, "skipped_unknown": 1, "skipped_both": 1}
    assert len(list((tmp_path / "work/normal").iterdir())) == 11
    assert len(list((tmp_path / "work/anomalous").iterdir())) == 4
    link = next((tmp_path / "work/normal").iterdir())
    assert link.is_symlink() and link.suffix == ".jpg"


def test_too_few_normal(tmp_path):
    with pytest.raises(ValueError, match="at least 10 images tagged 'normal'"):
        prepare(write_jsonl(tmp_path, [["normal"]] * 9), tmp_path / "work")


def test_few_anomalous_are_ignored(tmp_path):
    counts = prepare(write_jsonl(tmp_path, [["normal"]] * 10 + [["anomalous"]] * 3), tmp_path / "work")
    assert counts["anomalous"] == 0
    assert not any((tmp_path / "work/anomalous").iterdir())


def test_extensionless_paths_get_image_suffix(tmp_path):
    rows = []
    for i in range(10):
        img = tmp_path / f"blob{i}"
        img.write_bytes(b"\x89PNG\r\n" if i % 2 else b"\xff\xd8\xff")
        rows.append(json.dumps({"image_path": str(img), "classification_annotations": [{"annotation_label": "normal"}]}))
    (tmp_path / "d.jsonl").write_text("\n".join(rows) + "\n")
    prepare(tmp_path / "d.jsonl", tmp_path / "work")
    assert sorted(p.suffix for p in (tmp_path / "work/normal").iterdir()) == [".jpg"] * 5 + [".png"] * 5
