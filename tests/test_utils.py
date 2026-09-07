import numpy as np
import pandas as pd

from src.metrics import best_threshold, compute_metrics
from src.train import make_param_grid
from src.utils import clean_adult, file_fingerprint, read_meta, split_data, write_meta


def test_clean_adult_normalises_target_and_missing():
    df = pd.DataFrame(
        {
            "workclass": [" Private", " ?", "Self-emp "],
            "age": [30, 40, 50],
            "income": [" >50K", " <=50K.", ">50K."],
        }
    )
    out = clean_adult(df, target="income", positive_label=">50K")
    assert out["income"].tolist() == [1, 0, 1]
    assert out["workclass"].tolist()[0] == "Private"
    assert pd.isna(out.loc[1, "workclass"])
    assert out["age"].dtype == "float64"  # numeric features are cast to float for the signature


def test_split_is_stratified_and_reproducible(toy_data, toy_cfg):
    X, y = toy_data
    a = split_data(X, y, toy_cfg)
    b = split_data(X, y, toy_cfg)
    pd.testing.assert_frame_equal(a[1], b[1])
    assert np.isclose(a[2].mean(), a[3].mean(), atol=0.05)


def test_make_param_grid_prefixes_and_wraps_scalars():
    grid = make_param_grid({"C": [0.1, 1], "penalty": "l2"})
    assert grid == {"model__C": [0.1, 1], "model__penalty": ["l2"]}
    assert make_param_grid({"C": [1]}, "model__estimator__") == {"model__estimator__C": [1]}


def test_best_threshold_beats_default_on_skewed_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    score = np.clip(0.25 * y + rng.normal(0.15, 0.08, 2000), 0, 1)  # positives around 0.4
    t, f1 = best_threshold(y, score)
    m = compute_metrics(y, score, t)
    assert 0 < t < 0.5
    assert m["test_f1"] >= m["test_f1_at_0.5"]
    assert np.isclose(m["test_f1"], f1, atol=1e-9)
    assert set(m) >= {"test_roc_auc", "test_brier", "test_average_precision", "decision_threshold"}


def test_fingerprint_and_meta_roundtrip(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    fp = file_fingerprint(p)
    assert fp["data_bytes"] == 8 and len(fp["data_sha256"]) == 16
    meta_file = tmp_path / "meta.json"
    write_meta(meta_file, {"version": "3", "threshold": np.float64(0.4), "path": p})
    assert read_meta(meta_file) == {"version": "3", "threshold": 0.4, "path": str(p)}
    assert read_meta(tmp_path / "missing.json") == {}
