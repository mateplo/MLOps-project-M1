"""Export the exported champion (artifacts/model.joblib) to ONNX for in-browser inference.

Produces artifacts/model.onnx + artifacts/preprocess.json (what the JavaScript side must do
before feeding the graph). Three sklearn features have no ONNX converter and are handled here:

- FunctionTransformer(log1p)       -> custom converter: Log(Add(x, 1))
- SimpleImputer on strings w/ NaN  -> the exported imputer treats the string "?" as missing
- OneHotEncoder(min_frequency=...) -> replaced by a plain encoder whose categories are the
  frequent ones + the "infrequent_sklearn" bucket; the JS maps rare/unknown values to that bucket

The export is validated against the sklearn pipeline on a sample of the training data.

Usage:
    python -m src.export_onnx --config configs/config.yaml [--check-rows 5000]
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.meta import META_FILENAME, read_meta
from src.utils import get_logger, load_config, resolve

log = get_logger(__name__)

INFREQUENT = "infrequent_sklearn"
MISSING = "?"
OPSET = 17


def _patch_onnx_bool_attributes() -> None:
    """skl2onnx emits numpy bools in `nodes_missing_value_tracks_true` (HGB); ONNX wants ints."""
    import onnx.helper as helper

    if getattr(helper, "_bool_patch", False):
        return
    original = helper.make_attribute

    def make_attribute(key, value, *args, **kwargs):
        if (
            isinstance(value, (list, tuple, np.ndarray))
            and len(value)
            and any(isinstance(v, (bool, np.bool_)) for v in value)
        ):
            value = [int(v) for v in value]
        return original(key, value, *args, **kwargs)

    helper.make_attribute = make_attribute
    helper._bool_patch = True


def _register_log1p_converter() -> None:
    from skl2onnx import update_registered_converter
    from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxLog
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.preprocessing import FunctionTransformer

    def shape(operator):
        operator.outputs[0].type = FloatTensorType(operator.inputs[0].type.shape)

    def convert(scope, operator, container):
        one = np.array([1.0], dtype=np.float32)
        OnnxLog(
            OnnxAdd(operator.inputs[0], one, op_version=OPSET),
            op_version=OPSET,
            output_names=[operator.outputs[0].full_name],
        ).add_to(scope, container)

    update_registered_converter(
        FunctionTransformer, "AdultLog1pTransformer", shape, convert, overwrite=True
    )


def prepare_for_export(pipe):
    """Return (export_pipeline, spec): a deep copy with export-only substitutions + the JS spec."""
    from sklearn.preprocessing import OneHotEncoder

    pipe = copy.deepcopy(pipe)
    pre = pipe.named_steps["pre"]
    branches = {name: list(cols) for name, _, cols in pre.transformers_ if name != "remainder"}
    numeric = branches.get("num", [])
    log1p = branches.get("log", [])
    categorical = branches.get("cat", [])

    cat_pipe = pre.named_transformers_["cat"]
    cat_pipe.named_steps["imputer"].missing_values = MISSING
    ohe = cat_pipe.named_steps["ohe"]
    frequent: dict[str, list[str]] = {}
    export_categories = []
    infrequent_attr = getattr(ohe, "infrequent_categories_", [None] * len(categorical))
    for i, col in enumerate(categorical):
        infreq = set(infrequent_attr[i]) if infrequent_attr[i] is not None else set()
        freq = [str(c) for c in ohe.categories_[i] if c not in infreq]
        frequent[col] = freq
        export_categories.append(freq + ([INFREQUENT] if infreq else []))
    plain = OneHotEncoder(
        categories=export_categories, handle_unknown="ignore", sparse_output=False
    )
    plain.fit(pd.DataFrame({c: [export_categories[i][0]] for i, c in enumerate(categorical)}))
    if list(plain.get_feature_names_out(categorical)) != list(
        ohe.get_feature_names_out(categorical)
    ):
        raise RuntimeError("one-hot column order mismatch between original and export encoder")
    cat_pipe.steps[-1] = ("ohe", plain)

    spec = {
        "numeric": numeric + log1p,  # all fed as float32 [n,1]; log1p is inside the graph
        "categorical": categorical,
        "frequent_categories": frequent,
        "missing_token": MISSING,
        "infrequent_token": INFREQUENT,
        "input_order": numeric + log1p + categorical,
        "opset": OPSET,
    }
    return pipe, spec


def to_onnx(pipe, spec: dict) -> bytes:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    _patch_onnx_bool_attributes()
    _register_log1p_converter()
    inputs = [(c, FloatTensorType([None, 1])) for c in spec["numeric"]]
    inputs += [(c, StringTensorType([None, 1])) for c in spec["categorical"]]
    onx = convert_sklearn(
        pipe,
        initial_types=inputs,
        target_opset=OPSET,
        options={id(pipe.named_steps["model"]): {"zipmap": False}},
    )
    return onx.SerializeToString()


def browser_preprocess(X: pd.DataFrame, spec: dict) -> dict:
    """Reference implementation of what app.js does; used to validate the export."""
    feed = {}
    for c in spec["numeric"]:
        feed[c] = (
            pd.to_numeric(X[c], errors="coerce").fillna(np.nan).to_numpy(np.float32).reshape(-1, 1)
        )
    for c in spec["categorical"]:
        s = X[c].astype(object).where(X[c].notna(), spec["missing_token"]).astype(str).str.strip()
        s = s.where(s != "", spec["missing_token"])
        allowed = set(spec["frequent_categories"][c]) | {spec["missing_token"]}
        s = s.where(s.isin(allowed), spec["infrequent_token"])
        feed[c] = s.to_numpy().astype(object).reshape(-1, 1)
    return feed


def validate(onnx_bytes: bytes, pipe, X: pd.DataFrame, spec: dict, tol: float = 1e-4) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_bytes)
    probs = sess.run(None, browser_preprocess(X, spec))[1][:, 1]
    ref = pipe.predict_proba(X)[:, 1]
    diff = float(np.max(np.abs(ref - probs)))
    if diff > tol:
        raise RuntimeError(f"ONNX export diverges from sklearn: max |diff| = {diff:.2e} > {tol}")
    return diff


def export_onnx(cfg: dict, check_rows: int = 5000) -> Path:
    art_dir = resolve(cfg["artifacts"]["dir"])
    model_path = resolve(cfg["artifacts"]["model_path"])
    pipe = joblib.load(model_path)
    meta = read_meta(art_dir / META_FILENAME)

    export_pipe, spec = prepare_for_export(pipe)
    onnx_bytes = to_onnx(export_pipe, spec)

    csv = resolve(cfg["data"]["csv_path"])
    if csv.exists() and check_rows > 0:
        from src.utils import load_dataframe

        df = load_dataframe(cfg, validate=False)
        X = df[spec["input_order"]].sample(min(check_rows, len(df)), random_state=0)
        diff = validate(onnx_bytes, pipe, X, spec)
        log.info("validated on %d rows: max |sklearn - onnx| = %.2e", len(X), diff)
    else:
        log.warning("no training data found: export not validated against sklearn")

    out = art_dir / "model.onnx"
    out.write_bytes(onnx_bytes)
    spec.update(
        {
            "decision_threshold": float(meta.get("decision_threshold", 0.5)),
            "model_name": meta.get("model_name"),
            "version": meta.get("version"),
            "run_id": meta.get("run_id"),
            "model_type": meta.get("model_type"),
            "labels": {"0": "<=50K", "1": ">50K"},
        }
    )
    (art_dir / "preprocess.json").write_text(json.dumps(spec, indent=2))
    log.info("wrote %s (%d KB) and preprocess.json", out, len(onnx_bytes) // 1024)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export artifacts/model.joblib to ONNX")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--check-rows", type=int, default=5000)
    args = parser.parse_args()
    export_onnx(load_config(args.config), args.check_rows)


if __name__ == "__main__":
    main()
