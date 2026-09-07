PYTHON  ?= python3.11
PY      ?= .venv/bin/python
PIP     ?= .venv/bin/pip
CONFIG  ?= configs/config.yaml
EXP     ?= adult-income
PORT    ?= 8000
# 5000 is taken by the AirPlay Receiver on macOS
MLFLOW_PORT ?= 5001
IMAGE   ?= adult-income-classifier:latest
IMAGE_TRAIN ?= adult-income-classifier:train

.PHONY: help init data train evaluate all predict test lint format ui serve build build-train docker-train docker-serve clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

init:            ## Create venv (Python 3.11 by default: make init PYTHON=python3.12) and install deps
	$(PYTHON) -m venv .venv && $(PIP) install -U pip && $(PIP) install -r requirements.txt

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

ui:              ## Launch the MLflow UI (http://127.0.0.1:$(MLFLOW_PORT))
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port $(MLFLOW_PORT)

serve:           ## Run the FastAPI microservice locally
	.venv/bin/uvicorn src.app:app --host 0.0.0.0 --port $(PORT) --reload

build:           ## Build the serving Docker image (minimal runtime)
	docker build --target serve -t $(IMAGE) .

build-train:     ## Build the training Docker image (MLflow + plots)
	docker build --target train -t $(IMAGE_TRAIN) .

docker-train: build-train  ## Train inside Docker. The project is mounted at its host path so MLflow's absolute artifact URIs stay valid
	docker run --rm --user $$(id -u):$$(id -g) -e HOME=/tmp \
		-v "$(PWD):$(PWD)" -w "$(PWD)" \
		$(IMAGE_TRAIN) python -m src.train --config $(CONFIG)

docker-serve:    ## Serve the API from Docker on $(PORT)
	docker run --rm -p $(PORT):8000 -v "$(PWD)/artifacts:/app/artifacts" $(IMAGE)

clean:           ## Remove caches and generated artifacts (keeps data + mlflow.db)
	rm -rf .pytest_cache .ruff_cache artifacts/* && find . -name __pycache__ -type d -exec rm -rf {} +
