import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils import PROJECT_ROOT

NUMERIC = ["age", "hours_per_week"]
CATEGORICAL = ["workclass", "sex"]

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "hours_per_week": rng.integers(10, 80, n).astype(float),
            "workclass": rng.choice(["Private", "Self-emp", None], n),
            "sex": rng.choice(["Male", "Female"], n),
        }
    )
    y = pd.Series(((X["age"] > 40) & (X["hours_per_week"] > 35)).astype(int))
    return X, y


@pytest.fixture
def toy_cfg():
    return {
        "data": {
            "target": "income",
            "positive_label": ">50K",
            "test_size": 0.25,
            "random_state": 0,
        },
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
    }


def make_adult_like(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataframe with the raw UCI Adult schema ('?' missing, '>50K.' variants)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(17, 90, n)
    edu = rng.integers(1, 17, n)
    hours = rng.integers(1, 99, n)
    gain = np.where(rng.random(n) < 0.1, rng.integers(1000, 50000, n), 0)
    logit = -6 + 0.05 * age + 0.3 * edu + 0.02 * hours + (gain > 0) * 1.5
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(
        {
            "age": age,
            "workclass": rng.choice(
                ["Private", "Self-emp-inc", "State-gov", "?"], n, p=[0.7, 0.15, 0.1, 0.05]
            ),
            "fnlwgt": rng.integers(10_000, 500_000, n),
            "education": rng.choice(["Bachelors", "HS-grad", "Masters"], n),
            "education_num": edu,
            "marital_status": rng.choice(["Married-civ-spouse", "Never-married", "Divorced"], n),
            "occupation": rng.choice(
                ["Exec-managerial", "Sales", "Craft-repair", "?"], n, p=[0.3, 0.3, 0.35, 0.05]
            ),
            "relationship": rng.choice(["Husband", "Not-in-family", "Wife"], n),
            "race": rng.choice(["White", "Black", "Asian-Pac-Islander"], n, p=[0.8, 0.15, 0.05]),
            "sex": rng.choice(["Male", "Female"], n),
            "capital_gain": gain,
            "capital_loss": np.where(rng.random(n) < 0.05, rng.integers(100, 3000, n), 0),
            "hours_per_week": hours,
            "native_country": rng.choice(
                ["United-States", "Mexico", "Rare-Land"], n, p=[0.9, 0.08, 0.02]
            ),
            "income": np.where(
                y, rng.choice([">50K", ">50K."], n), rng.choice(["<=50K", "<=50K."], n)
            ),
        }
    )
    return df[COLUMNS]


@pytest.fixture
def adult_like_df():
    return make_adult_like()


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Throw-away project: synthetic raw.csv, config with tmp paths, sqlite MLflow in tmp."""
    df = make_adult_like()
    csv = tmp_path / "raw.csv"
    df.to_csv(csv, index=False)

    cfg = copy.deepcopy(yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text()))
    cfg["data"]["csv_path"] = str(csv)
    cfg["artifacts"] = {
        "dir": str(tmp_path / "artifacts"),
        "model_path": str(tmp_path / "artifacts" / "model.joblib"),
    }
    cfg["model"] = {"type": "logreg", "params": {"C": [1.0]}}
    cfg["cv"] = {"strategy": "StratifiedKFold", "n_splits": 2, "scoring": "roc_auc", "n_jobs": 1}
    cfg["validation"]["positive_rate"] = [0.05, 0.95]
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-exp")
    monkeypatch.chdir(tmp_path)
    return {"cfg": cfg, "cfg_path": cfg_path, "dir": tmp_path, "df": df}
