"""artifacts/model_meta.json: the contract between the registry and serving.

Kept free of heavy imports so the lightweight API image can use it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

META_FILENAME = "model_meta.json"


def read_meta(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def write_meta(path: str | Path, meta: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, default=_jsonable))


def _jsonable(v):
    try:
        import numpy as np

        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
    except ImportError:  # pragma: no cover
        pass
    return str(v)
