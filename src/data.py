"""Download the UCI Adult Income dataset and write it as a headed CSV to data/raw.csv.

Both official files (adult.data + adult.test) are concatenated so that the
train/test split is fully controlled by the config (stratified, seeded).
"""

from __future__ import annotations

import argparse
import io

import pandas as pd
import requests

from src.utils import PROJECT_ROOT, load_config


def _fetch(url: str, columns: list[str], skiprows: int = 0) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(
        io.StringIO(resp.text),
        header=None,
        names=columns,
        skiprows=skiprows,
        skipinitialspace=True,
        na_values=["?"],
    )


def download(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    cols = d["columns"]
    train = _fetch(d["url"], cols)
    # adult.test has a junk first line ("|1x3 Cross validator")
    test = _fetch(d["test_url"], cols, skiprows=1)
    df = pd.concat([train, test], ignore_index=True).dropna(subset=[d["target"]])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Adult Income dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = PROJECT_ROOT / cfg["data"]["csv_path"]
    if out.exists() and not args.force:
        print(f"{out} already exists, skipping (use --force to re-download).")
        return

    df = download(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols to {out}")


if __name__ == "__main__":
    main()
