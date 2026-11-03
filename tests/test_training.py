import json

from model.training import model_metrics


def test_model_metrics_is_lists_of_finite_floats():
    m = model_metrics({"train_loss": [0.9, float("nan"), 0.2]}, {"image_AUROC": 1.0, "image_F1Score": float("nan")})
    assert m == {"train_loss": [0.9, 0.2], "image_AUROC": [1.0]}
    assert "NaN" not in json.dumps(m)
