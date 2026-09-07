"""End-to-end: train -> promote -> evaluate -> export -> drift on a synthetic dataset,
with a throw-away sqlite MLflow backend. Checks runs, metrics, registry aliases and artifacts."""

import json

import mlflow
import pandas as pd
import pytest

from src.drift import drift_report
from src.evaluate import evaluate
from src.export import export
from src.promote import promote
from src.train import train
from src.utils import META_FILENAME, read_meta


@pytest.mark.slow
def test_full_pipeline(tmp_project):
    cfg, cfg_path, tmp = tmp_project["cfg"], tmp_project["cfg_path"], tmp_project["dir"]
    art = tmp / "artifacts"
    name = cfg["mlflow"]["registered_model_name"]

    # ---- train ---------------------------------------------------------------
    run_id = train(cfg, cfg_path)
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.tags["stage"] == "train"
    assert {
        "test_roc_auc",
        "test_f1",
        "test_brier",
        "cv_best_roc_auc",
        "cv_f1_at_threshold",
    } <= set(run.data.metrics)
    assert run.data.params["data_rows"] == "1500" and "data_sha256" in run.data.params
    assert run.inputs.dataset_inputs, "dataset lineage should be logged via log_input"
    assert (art / "model.joblib").exists()
    meta = read_meta(art / META_FILENAME)
    assert meta["run_id"] == run_id and 0 < meta["decision_threshold"] < 1
    assert meta["alias"] == "challenger"
    v1 = meta["version"]
    assert str(client.get_model_version_by_alias(name, "challenger").version) == v1
    assert client.get_model_version(name, v1).tags["decision_threshold"]

    # ---- promote: no champion -> promoted -------------------------------------
    res = promote(cfg)
    assert res["promoted"] is True and res["champion_version"] is None
    assert str(client.get_model_version_by_alias(name, "champion").version) == v1
    assert read_meta(art / META_FILENAME)["alias"] == "champion"

    # ---- second, worse model -> not promoted ----------------------------------
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["features"] = {"numeric": ["hours_per_week"], "categorical": ["race"]}  # weak features
    train(cfg2, cfg_path)
    v2 = read_meta(art / META_FILENAME)["version"]
    assert v2 != v1
    res2 = promote(cfg2)
    assert res2["promoted"] is False and str(res2["candidate_version"]) == v2
    assert str(client.get_model_version_by_alias(name, "champion").version) == v1
    assert str(client.get_model_version_by_alias(name, "challenger").version) == v2
    assert promote(cfg2, force=True)["promoted"] is True  # --force overrides

    # ---- export the champion + evaluate it --------------------------------------
    exported = export(cfg, alias="champion")
    assert exported["version"] == v2 and (art / "model.joblib").exists()
    metrics = evaluate(cfg, str(art / "model.joblib"))
    assert 0.5 < metrics["test_roc_auc"] <= 1.0
    for f in (
        "roc_curve.png",
        "pr_curve.png",
        "confusion_matrix.png",
        "calibration_curve.png",
        "predictions.csv",
        "subgroup_metrics.csv",
        "classification_report.json",
    ):
        assert (art / f).exists(), f
    sub = pd.read_csv(art / "subgroup_metrics.csv")
    assert set(sub["feature"]) == {"sex", "race"}
    metrics_registry = evaluate(cfg, f"models:/{name}@champion")
    assert metrics_registry["test_roc_auc"] == pytest.approx(metrics["test_roc_auc"])

    # ---- drift: log built from the training data itself => small PSI ------------
    log_path = tmp / "predictions.jsonl"
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    with open(log_path, "w") as f:
        for _, r in tmp_project["df"].sample(400, random_state=1).iterrows():
            f.write(
                json.dumps(
                    {
                        "features": {
                            k: (
                                None
                                if pd.isna(r[k])
                                else (int(r[k]) if k in cfg["features"]["numeric"] else r[k])
                            )
                            for k in feats
                        },
                        "score": 0.1,
                        "label": "<=50K",
                    }
                )
                + "\n"
            )
    rep = drift_report(cfg, log_path, min_rows=100)
    assert set(rep["feature"]) >= set(feats)
    assert (rep.loc[rep["type"] != "prediction", "psi"] < 0.1).all()

    # experiment timeline: train x2, promote x3, evaluate x2 (+ autolog children)
    exp = client.get_experiment_by_name("test-exp")
    stages = [
        r.data.tags.get("stage") for r in client.search_runs(exp.experiment_id, max_results=500)
    ]
    assert (
        stages.count("train") == 2
        and stages.count("promote") == 3
        and stages.count("evaluate") == 2
    )
