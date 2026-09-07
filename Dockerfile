FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db MODEL_PATH=artifacts/model.joblib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY examples ./examples

EXPOSE 8000
# Default: serve the API. Override the command to train:
#   docker run ... adult-income-classifier python -m src.train --config configs/config.yaml
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
