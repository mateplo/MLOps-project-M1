import numpy as np
import pandas as pd
import pytest

NUMERIC = ["age", "hours_per_week"]
CATEGORICAL = ["workclass", "sex"]


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "hours_per_week": rng.integers(10, 80, n),
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
