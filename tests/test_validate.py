import pytest
import yaml

from src.utils import PROJECT_ROOT
from src.validate import DataValidationError, validate_dataframe, validate_or_raise


@pytest.fixture
def cfg():
    c = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text())
    c["validation"]["positive_rate"] = [0.05, 0.95]
    return c


def test_valid_dataframe_passes(adult_like_df, cfg):
    assert validate_dataframe(adult_like_df, cfg) == []
    validate_or_raise(adult_like_df, cfg)


def test_missing_column_is_reported(adult_like_df, cfg):
    errors = validate_dataframe(adult_like_df.drop(columns=["occupation"]), cfg)
    assert len(errors) == 1 and "missing columns" in errors[0]


def test_out_of_range_and_bad_target(adult_like_df, cfg):
    df = adult_like_df.copy()
    df.loc[0, "age"] = 250
    df.loc[1, "income"] = "maybe"
    errors = validate_dataframe(df, cfg)
    assert any("age" in e for e in errors)
    assert any("unexpected target values" in e for e in errors)


def test_too_many_missing_and_too_few_rows(adult_like_df, cfg):
    df = adult_like_df.head(500).copy()
    df["workclass"] = "?"
    with pytest.raises(DataValidationError) as exc:
        validate_or_raise(df, cfg)
    msg = str(exc.value)
    assert "too few rows" in msg and "workclass: missing rate" in msg
