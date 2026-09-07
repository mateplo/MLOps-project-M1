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
N       ?= 300

.PHONY: help init lock data validate train promote evaluate export all predict test lint format ui serve \
        simulate simulate-drift drift build build-train docker-train docker-serve \
        compose-up compose-down compose-train compose-monitoring clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- environment
init:            ## Create venv (Python 3.11 by default: make init PYTHON=python3.12) and install deps
	$(PYTHON) -m venv .venv && $(PIP) install -U pip && $(PIP) install -r requirements.txt

lock:            ## Freeze exact versions into requirements.lock / requirements-serve.lock
	$(PIP) freeze --exclude-editable | grep -v "^-e" > requirements.lock
	rm -rf /tmp/serve-venv && $(PYTHON) -m venv /tmp/serve-venv && /tmp/serve-venv/bin/pip install -q -U pip \
		&& /tmp/serve-venv/bin/pip install -q -r requirements-serve.txt && /tmp/serve-venv/bin/pip freeze > requirements-serve.lock

# ---------------------------------------------------------------- ML pipeline
data:            ## Download the UCI Adult dataset to data/raw.csv
	$(PY) -m src.data --config $(CONFIG)

validate:        ## Validate data/raw.csv against the rules in the config
	$(PY) -m src.validate --config $(CONFIG)

train: data      ## Train + tune, track with MLflow, register as `challenger`
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) -m src.train --config $(CONFIG)

promote:         ## Promote the last trained version to `champion` if it beats the current one
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) -m src.promote --config $(CONFIG)

evaluate:        ## Final evaluation of artifacts/model.joblib (or MODEL=models:/AdultIncomeClassifier@champion)
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) -m src.evaluate --config $(CONFIG) $(if $(MODEL),--model $(MODEL),)

export:          ## Export the champion from the registry to artifacts/ (for the API image)
	$(PY) -m src.export --config $(CONFIG)

all: train promote evaluate  ## Full pipeline: train -> promote -> evaluate

predict:         ## Batch inference: make predict INPUT=file.csv OUTPUT=out.csv
	$(PY) -m src.predict --config $(CONFIG) --input $(INPUT) --output $(OUTPUT)

# ---------------------------------------------------------------- quality
test:            ## Run unit + end-to-end tests
	$(PY) -m pytest

lint:            ## Lint with ruff
	$(PY) -m ruff check .

format:          ## Auto-format / fix with ruff
	$(PY) -m ruff format . && $(PY) -m ruff check --fix .

# ---------------------------------------------------------------- serving & monitoring
ui:              ## Launch the MLflow UI (http://127.0.0.1:$(MLFLOW_PORT))
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port $(MLFLOW_PORT)

serve:           ## Run the FastAPI microservice locally (docs: http://localhost:$(PORT)/docs)
	.venv/bin/uvicorn src.app:app --host 0.0.0.0 --port $(PORT) --reload

simulate:        ## Send N real rows to the API (make simulate N=500)
	$(PY) -m src.simulate_traffic --config $(CONFIG) --url http://localhost:$(PORT) --n $(N)

simulate-drift:  ## Same, with a biased sample to make drift visible
	$(PY) -m src.simulate_traffic --config $(CONFIG) --url http://localhost:$(PORT) --n $(N) --drift

drift:           ## PSI drift report from logs/predictions.jsonl -> artifacts/drift_report.csv + MLflow run
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) -m src.drift --config $(CONFIG)

# ---------------------------------------------------------------- docker
build:           ## Build the serving Docker image (minimal runtime)
	docker build --target serve -t $(IMAGE) .

build-train:     ## Build the training Docker image (MLflow + plots)
	docker build --target train -t $(IMAGE_TRAIN) .

docker-train: build-train  ## Train inside Docker. The project is mounted at its host path so MLflow's absolute artifact URIs stay valid
	docker run --rm --user $$(id -u):$$(id -g) -e HOME=/tmp \
		-v "$(PWD):$(PWD)" -w "$(PWD)" \
		$(IMAGE_TRAIN) python -m src.train --config $(CONFIG)

docker-serve:    ## Serve the API from Docker on $(PORT)
	docker run --rm -p $(PORT):8000 -v "$(PWD)/artifacts:/app/artifacts" -v "$(PWD)/logs:/app/logs" $(IMAGE)

compose-up:      ## MLflow server (:$(MLFLOW_PORT)) + API (:$(PORT)) via docker compose
	docker compose up -d --build mlflow api

compose-monitoring: ## Also start Prometheus (:9090) + Grafana (:3000, admin/admin)
	docker compose --profile monitoring up -d --build

compose-train:   ## Run data -> train -> promote -> evaluate -> export against the MLflow server
	docker compose --profile train run --rm --build train

compose-down:    ## Stop everything (keeps volumes)
	docker compose --profile monitoring --profile train down

clean:           ## Remove caches and generated artifacts (keeps data + mlflow.db)
	rm -rf .pytest_cache .ruff_cache artifacts/* logs/* && find . -name __pycache__ -type d -exec rm -rf {} +
