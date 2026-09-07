#!/bin/sh
# Serving image entrypoint: optionally fetch the model from the HF Hub, then serve.
set -e
if [ -n "${MODEL_REPO:-}" ]; then
  python -m src.fetch_model
fi
exec uvicorn src.app:app --host 0.0.0.0 --port "${PORT:-8000}"
