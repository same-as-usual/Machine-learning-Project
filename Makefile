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

data-webis:  ## Webis Clickbait 2017 (graded intensity + article bodies)
	$(PY) -m manipulens.data.ingest_webis

train-transformer:  ## multi-task fine-tune (DeBERTa-v3-small by default; GPU recommended)
	$(PY) -m manipulens.models.transformer

train-transformer-smoke:  ## CPU-friendly smoke run (MiniLM, small subset)
	$(PY) -m manipulens.models.transformer --smoke --model-name sentence-transformers/all-MiniLM-L6-v2

distill:  ## teacher -> MiniLM student on soft labels
	$(PY) -m manipulens.models.distill

export-onnx:  ## student -> ONNX -> INT8 + parity/latency benchmark
	$(PY) -m manipulens.models.export_onnx --model-dir models/artifacts/student

label-validate:  ## LLM labels vs human gold set (needs ANTHROPIC_API_KEY)
	$(PY) -m manipulens.labeling.llm_labeler validate-gold

label:  ## LLM-label training headlines (budget-capped, cached)
	$(PY) -m manipulens.labeling.llm_labeler label

serve:
	.venv/bin/uvicorn manipulens.api.main:app --host 0.0.0.0 --port 8000

all: data train test test-behavioral

clean:
	rm -rf data/interim data/processed models/artifacts
