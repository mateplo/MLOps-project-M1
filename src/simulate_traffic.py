"""Send requests to the API from real rows to populate the prediction log / Prometheus.

`--drift` biases the sample (older, more educated, more capital gains) to make drift visible.

Usage:
    python -m src.simulate_traffic --n 500 [--url http://localhost:8000] [--drift]
"""

from __future__ import annotations

import argparse
import time

import httpx
import pandas as pd

from src.utils import get_logger, load_config, load_dataframe

log = get_logger(__name__)


def sample_requests(cfg: dict, n: int, drift: bool, seed: int = 0) -> list[dict]:
    df = load_dataframe(cfg, validate=False)
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    if drift:
        weights = (
            (df["age"] >= 45).astype(float) * 3 + (df["education_num"] >= 13).astype(float) * 3 + 1
        )
        df = df.sample(n=n, weights=weights, random_state=seed)
        df["capital_gain"] = df["capital_gain"] * 2
        df["hours_per_week"] = (df["hours_per_week"] + 10).clip(upper=99)
    else:
        df = df.sample(n=n, random_state=seed)
    records = []
    for _, r in df[feats].iterrows():
        rec = {}
        for k, v in r.items():
            if pd.isna(v):
                continue
            rec[k] = int(v) if k in cfg["features"]["numeric"] else v
        records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate API traffic")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    records = sample_requests(cfg, args.n, args.drift, args.seed)
    t0 = time.perf_counter()
    ok = errors = positives = 0
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for rec in records:
            resp = client.post("/predict", json=rec)
            if resp.status_code == 200:
                ok += 1
                positives += resp.json()["label"] == ">50K"
            else:
                errors += 1
    log.info(
        "%d ok, %d errors, %.1f%% predicted >50K, %.1fs (drift=%s)",
        ok,
        errors,
        100 * positives / max(ok, 1),
        time.perf_counter() - t0,
        args.drift,
    )


if __name__ == "__main__":
    main()
