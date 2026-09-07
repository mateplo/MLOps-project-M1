PY      ?= .venv/bin/python
PIP     ?= .venv/bin/pip
CONFIG  ?= configs/config.yaml
EXP     ?= adult-income
PORT    ?= 8000
IMAGE   ?= adult-income-classifier:latest

.PHONY: help init data train evaluate all predict test lint format ui serve build docker-train docker-serve clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

init:            ## Create venv and install dependencies
	python3 -m venv .venv && $(PIP) install -U pip && $(PIP) install -r requirements.txt

data:            ## Download the UCI Adult dataset to data/raw.csv
	$(PY) -m src.data --config $(CONFIG)

train: data      ## Train + tune, track with MLflow, register the best model
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) -m src.train --config $(CONFIG)

evaluate:        ## Final evaluation: plots, predictions CSV, subgroup metrics -> MLflow
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) -m src.evaluate --config $(CONFIG)

all: train evaluate  ## Full pipeline

predict:         ## Batch inference: make predict INPUT=file.csv OUTPUT=out.csv
	$(PY) -m src.predict --config $(CONFIG) --input $(INPUT) --output $(OUTPUT)

test:            ## Run unit tests
	$(PY) -m pytest

lint:            ## Lint with ruff
	$(PY) -m ruff check .

format:          ## Auto-format / fix with ruff
	$(PY) -m ruff format . && $(PY) -m ruff check --fix .

ui:              ## Launch the MLflow UI (http://127.0.0.1:5000)
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:           ## Run the FastAPI microservice locally
	.venv/bin/uvicorn src.app:app --host 0.0.0.0 --port $(PORT) --reload

build:           ## Build the Docker image
	docker build -t $(IMAGE) .

docker-train:    ## Train inside Docker (mounts data/, artifacts/ and mlflow.db)
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/artifacts:/app/artifacts \
		-v $(PWD)/mlruns:/app/mlruns -v $(PWD)/mlflow.db:/app/mlflow.db \
		$(IMAGE) python -m src.train --config $(CONFIG)

docker-serve:    ## Serve the API from Docker on $(PORT)
	docker run --rm -p $(PORT):8000 -v $(PWD)/artifacts:/app/artifacts $(IMAGE)

clean:           ## Remove caches and generated artifacts (keeps data + mlflow.db)
	rm -rf .pytest_cache .ruff_cache artifacts/* && find . -name __pycache__ -type d -exec rm -rf {} +
