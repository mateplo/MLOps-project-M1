import numpy as np
import pandas as pd

from src.train import make_param_grid
from src.utils import clean_adult, split_data


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
    assert out["age"].tolist() == [30, 40, 50]


def test_split_is_stratified_and_reproducible(toy_data, toy_cfg):
    X, y = toy_data
    a = split_data(X, y, toy_cfg)
    b = split_data(X, y, toy_cfg)
    pd.testing.assert_frame_equal(a[1], b[1])
    assert np.isclose(a[2].mean(), a[3].mean(), atol=0.05)


def test_make_param_grid_prefixes_and_wraps_scalars():
    grid = make_param_grid({"C": [0.1, 1], "penalty": "l2"})
    assert grid == {"model__C": [0.1, 1], "model__penalty": ["l2"]}
