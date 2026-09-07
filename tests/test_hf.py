import json

import src.fetch_model as fm
from src.deploy_space import read_env_file, render_dockerfile
from src.publish import model_card

META = {
    "model_name": "AdultIncomeClassifier",
    "version": "9",
    "alias": "champion",
    "run_id": "8eec6466abcd",
    "model_type": "hgb",
    "decision_threshold": 0.381,
    "data_sha256": "aa42",
    "trained_at": "2026-09-07",
    "metrics": {"test_roc_auc": 0.9303, "test_f1": 0.7284, "decision_threshold": 0.381},
}


def test_model_card_lists_identity_and_metrics():
    card = model_card(META)
    assert card.startswith("---\nlibrary_name: sklearn")
    assert "| Registry version | **9** (champion) |" in card
    assert "| test_roc_auc | 0.9303 |" in card
    assert "| decision_threshold |" not in card.split("## Held-out")[1]


def test_render_dockerfile_and_env(tmp_path):
    env_file = tmp_path / "space.env"
    env_file.write_text("# comment\nMODEL_REVISION=v9\n\nFOO = bar\n")
    env = read_env_file(env_file)
    assert env == {"MODEL_REVISION": "v9", "FOO": "bar"}
    df = render_dockerfile("ghcr.io/o/r:1.0.0", "o/model", env)
    assert 'FROM ghcr.io/o/r:1.0.0\nENV MODEL_REPO="o/model"\nENV MODEL_REVISION="v9"' in df


def test_fetch_skips_when_model_present(tmp_path, monkeypatch):
    (tmp_path / "model.joblib").write_bytes(b"x")
    (tmp_path / "model_meta.json").write_text(json.dumps({"version": "9"}))
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kw: (_ for _ in ()).throw(AssertionError("must not download")),
    )
    assert fm.fetch("o/m", "main", tmp_path)["version"] == "9"


def test_fetch_downloads_both_files(tmp_path, monkeypatch):
    src = tmp_path / "hub"
    src.mkdir()
    (src / "model.joblib").write_bytes(b"model")
    (src / "model_meta.json").write_text(json.dumps({"version": "10"}))
    calls = []

    def fake_download(repo_id, filename, revision, repo_type):
        calls.append((repo_id, filename, revision))
        return str(src / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    dest = tmp_path / "artifacts"
    meta = fm.fetch("o/m", "v10", dest)
    assert meta["version"] == "10" and (dest / "model.joblib").read_bytes() == b"model"
    assert calls == [("o/m", "model.joblib", "v10"), ("o/m", "model_meta.json", "v10")]
