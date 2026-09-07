import numpy as np
import pandas as pd

from src.drift import psi_categorical, psi_numeric


def test_psi_numeric_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(40, 10, 5000))
    same = pd.Series(rng.normal(40, 10, 5000))
    shifted = pd.Series(rng.normal(60, 10, 5000))
    assert psi_numeric(ref, same) < 0.05
    assert psi_numeric(ref, shifted) > 0.5


def test_psi_categorical_handles_new_and_missing_categories():
    ref = pd.Series(["a"] * 80 + ["b"] * 20)
    same = pd.Series(["a"] * 40 + ["b"] * 10)
    new = pd.Series(["c"] * 50 + [None] * 10)
    assert psi_categorical(ref, same) < 0.01
    assert psi_categorical(ref, new) > 1.0
