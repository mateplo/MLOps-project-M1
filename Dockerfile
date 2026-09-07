# syntax=docker/dockerfile:1
# Multi-stage build.
#   serve (default) : minimal runtime for the FastAPI microservice (no MLflow, no matplotlib, no dev tools)
#   train           : full environment for src.train / src.evaluate (MLflow tracking + plots)
#
#   docker build -t adult-income-classifier:latest .                  # serve image
#   docker build --target train -t adult-income-classifier:train .    # train image

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
# Stage 1: build a self-contained venv for serving
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder-serve
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt \
    && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /opt/venv -type d -name "tests" -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Stage 2: build a full venv for training (serve deps + MLflow, matplotlib...)
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder-train
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Common runtime base (non-root user, no pip, no build tools)
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime-base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=artifacts/model.joblib
RUN useradd --create-home --uid 1000 app && mkdir -p /app/artifacts /app/data && chown -R app:app /app
COPY --chown=app:app configs ./configs
COPY --chown=app:app src ./src
COPY --chown=app:app examples ./examples

# ---------------------------------------------------------------------------
# Target: train
# ---------------------------------------------------------------------------
FROM runtime-base AS train
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
COPY --from=builder-train /opt/venv /opt/venv
USER app
CMD ["python", "-m", "src.train", "--config", "configs/config.yaml"]

# ---------------------------------------------------------------------------
# Target: serve (default, last stage)
# ---------------------------------------------------------------------------
FROM runtime-base AS serve
COPY --from=builder-serve /opt/venv /opt/venv
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
