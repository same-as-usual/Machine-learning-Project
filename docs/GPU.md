# GPU fine-tune runbook

The committed transformer/distill artifacts are **CPU smoke runs** (MiniLM, 256
samples) that prove the pipeline. This runbook produces the real models. Any
single GPU with ≥8 GB VRAM is plenty — DeBERTa-v3-small is 142M params and the
inputs are 64-token headlines.

**Expected wall-clock on a T4 (free Colab):** fine-tune ~15–25 min,
distillation ~20–30 min, export ~2 min.

## Option A — any CUDA machine

```bash
git clone <repo> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu121  # match your CUDA
pip install transformers sentencepiece onnx onnxscript onnxruntime

# data (once)
make data data-webis

# optional but recommended: validated LLM taxonomy labels
export ANTHROPIC_API_KEY=...
make label-validate          # gates dimensions by gold-set alpha
make label                   # budget-capped labeling of the training set
# train() picks these up automatically and logs:
#   "validated LLM labels: N headlines ..., usable dims: [...]"

# the chain
make train-transformer       # DeBERTa-v3-small multi-task (params.yaml: transformer.*)
make distill                 # -> MiniLM student on teacher soft targets
make export-onnx             # -> ONNX -> INT8 + parity/latency benchmark
python apps/extension/build.py            # bundle the extension
python tools/test_tokenizer_parity.py     # JS/Python preprocessing parity (needs node)
```

## Option B — Google Colab (free T4)

One cell:

```python
!git clone <repo> project && cd project && pip install -q -e ".[dev]" transformers sentencepiece onnx onnxscript onnxruntime
%cd project
!python -m manipulens.data.ingest && python -m manipulens.data.dedup && python -m manipulens.data.splits
!python -m manipulens.data.ingest_webis
!python -m manipulens.models.transformer            # cuda picked up automatically
!python -m manipulens.models.distill
!python -m manipulens.models.export_onnx --model-dir models/artifacts/student
```

Then download `models/artifacts/student/` (Colab file pane or
`google.colab.files.download`) and drop it into the repo before running
`apps/extension/build.py` locally.

## Acceptance criteria (from the project plan)

| Check | Target | Where |
|---|---|---|
| Intensity Spearman vs Webis truthMean | ≥ 0.6 (leaderboard-competitive ~0.7) | `reports/transformer_eval.json` |
| Binary ROC-AUC | ≥ baseline's 0.9957 or explain gap | same |
| Student quality drop vs teacher | ≤ 2–3 pts | `reports/distill_eval.json` |
| INT8 size | < 30 MB | `reports/onnx_benchmark.json` |
| INT8 parity vs torch | max diff < 0.05 | same |
| Behavioral gate | all pass | `pytest -m behavioral` |
| JS/Python preprocessing parity | 10/10 probes | `tools/test_tokenizer_parity.py` |

## Tuning notes

- `params.yaml → transformer.*` holds LR/batch/epochs; the loop uses linear
  warmup+decay and grad clipping. On a T4, `batch_size: 64` fits easily.
- Scale `webis.archive` up to `clickbait17-train-170630.zip` (19,538 posts,
  937 MB) once disk allows — 8x more intensity supervision.
- If CPU-bound anywhere, `MANIPULENS_TORCH_THREADS` caps torch threads
  (oversubscription on shared cores is a 25x slowdown; see cap_torch_threads).
- After retraining, rerun `pytest -m behavioral` — the promotion gate
  (entity-swap invariance, directional lifts) must pass before the artifact
  ships; then rebuild the extension bundle.
