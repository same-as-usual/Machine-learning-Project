# ManipuLens

**Detects manipulation *techniques* in news headlines — not truth.**

A headline can't be fact-checked from twelve words. But the *techniques* used to manipulate you — curiosity gaps, outrage bait, fear-mongering, false certainty — are detectable, measurable, and explainable. ManipuLens scores headlines on a multi-dimensional manipulation taxonomy with **calibrated** probabilities and highlighted trigger phrases.

## Why this is different

1. **Multi-dimensional taxonomy, not binary clickbait** — six manipulation techniques defined in a written [annotation codebook](src/manipulens/labeling/codebook.md), with inter-annotator agreement (Krippendorff's α) gating which dimensions ship.
2. **Headline↔body "delivery gap"** *(roadmap)* — NLI cross-encoder checks whether the headline overpromises relative to the article body.
3. **In-browser inference** *(roadmap)* — distilled INT8 ONNX model (<30 MB) runs inside the browser extension; your browsing never leaves your machine.
4. **Published political-neutrality audit** *(roadmap)* — score parity across outlet leanings + entity-counterfactual tests, shipped as a report.

## Engineering rigor

- **Leakage-safe evaluation**: MinHash near-duplicate dedup + grouped splits (`src/manipulens/data/`). Random splits on headline data silently leak near-dupes and inflate every metric.
- **Calibrated scores**: isotonic/sigmoid calibration, ECE reported (`src/manipulens/models/calibrate.py`). A user-facing score must be calibrated or it's decoration.
- **Behavioral test suite as CI promotion gate** (`tests/behavioral/`): entity swaps must not move scores; prepending clickbait phrases must raise them.
- **Data contracts**: pandera schemas validated in CI (`src/manipulens/data/schemas.py`).
- **Weak supervision**: lexicon/pattern labeling functions double as explanation spans (`src/manipulens/labeling/`).

## Quickstart

```bash
make install        # venv + deps
make data           # ingest -> dedup -> leakage-safe splits
make train          # baselines + calibration + eval report
make test           # unit + data-contract tests
make test-behavioral  # model promotion gate
make serve          # FastAPI on :8000
```

Score a headline:

```bash
curl -s -X POST localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"headline": "You Won'\''t Believe What This Senator Said Next"}' | python3 -m json.tool
```

## Repo map

```
src/manipulens/
├── data/        # ingest, MinHash dedup, leakage-safe splits, pandera schemas
├── labeling/    # codebook.md, lexicons/, labeling functions, Krippendorff's α
├── models/      # baselines (TF-IDF+LogReg, LightGBM), calibration
├── eval/        # metrics (ECE, PR-AUC)
└── api/         # FastAPI service
tests/           # unit/, data/, behavioral/ (promotion gate)
docs/decisions/  # ADRs
reports/         # eval report, model card (generated)
```

## Roadmap

Phases 3–7 of the [project plan](docs/PLAN.md): DeBERTa-v3-small multi-task fine-tune → distillation + INT8 ONNX → delivery-gap NLI → web demo + MV3 extension with in-browser inference → neutrality & robustness audits → Hugging Face Hub release.

## Limitations

ManipuLens estimates the presence of *rhetorical techniques*. It does **not** verify factual accuracy, and a high manipulation score does not mean a story is false. See the model card (forthcoming) for details.
