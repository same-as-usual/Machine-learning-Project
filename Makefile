.PHONY: install lint typecheck test test-behavioral data train serve all clean

PY := .venv/bin/python
PIP := .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install -e ".[dev]"

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

format:
	.venv/bin/ruff format src tests
	.venv/bin/ruff check --fix src tests

typecheck:
	.venv/bin/mypy src

test:
	.venv/bin/pytest -m "not behavioral"

test-behavioral:  ## model promotion gate — requires a trained model artifact
	.venv/bin/pytest -m behavioral

data:  ## ingest -> dedup -> split
	$(PY) -m manipulens.data.ingest
	$(PY) -m manipulens.data.dedup
	$(PY) -m manipulens.data.splits

train:  ## train + calibrate baselines, write eval report
	$(PY) -m manipulens.models.baselines

serve:
	.venv/bin/uvicorn manipulens.api.main:app --host 0.0.0.0 --port 8000

all: data train test test-behavioral

clean:
	rm -rf data/interim data/processed models/artifacts
