<div align="center">

<img src="apps/extension/assets/icons/icon128.png" width="96" alt="ManipuLens icon">

# ManipuLens

**Detects manipulation *techniques* in news headlines — not truth.**

*A 23 MB model that runs entirely inside your browser, trained to spot curiosity gaps,
outrage bait, and fear-mongering — with political neutrality guaranteed by construction.*

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-81%20passing-brightgreen)
![Model](https://img.shields.io/badge/INT8%20model-22.9%20MB-orange)
![Privacy](https://img.shields.io/badge/inference-in--browser-purple)

</div>

---

A headline can't be fact-checked from twelve words. But the *techniques* used to
manipulate you — curiosity gaps, outrage bait, fear-mongering, false certainty — are
detectable, measurable, and explainable. ManipuLens scores headlines on a
multi-dimensional manipulation taxonomy with **calibrated** probabilities and
highlighted trigger phrases, and ships as a Chrome extension whose model
**never sends your browsing anywhere**.

## Results

| Model | Binary ROC-AUC | Intensity Spearman | Size | Notes |
|---|---|---|---|---|
| Rules only (lexicons) | 0.69 | — | — | floor |
| LightGBM + engineered features | 0.9957 | — | ~1 MB | calibrated, ECE 0.0048 |
| RoBERTa-base multi-task (teacher) | **0.9974** | **0.657** | 499 MB | GPU fine-tune |
| MiniLM student (distilled) | 0.9958 | 0.649 | 90 MB | −0.8 pt vs teacher |
| **INT8 ONNX student (ships)** | ≈ student | ≈ student | **22.9 MB** | max quant. error 0.027, ~16 ms/headline |

Evaluated on leakage-safe held-out splits of 31,993 labeled headlines
([Chakraborty 2016](https://github.com/bhargaviparanjape/clickbait)) and 2,459
crowd-rated posts with graded intensity + article bodies
([Webis Clickbait 2017](https://zenodo.org/records/5530410)).

## The four pillars

**1. Technique taxonomy, not binary clickbait.** Six manipulation dimensions —
curiosity gap, outrage bait, fear-mongering, false certainty, emotional framing,
sensational formatting — defined in a written [annotation codebook](src/manipulens/labeling/codebook.md)
with 0–2 ordinal scales, worked examples, and Krippendorff's-α agreement gating
which dimensions are allowed to ship.

**2. Headline↔body delivery gap.** A zero-shot NLI cross-encoder checks whether the
article *delivers what its headline promises* — the honest version of "lie detection."
Both NLI channels are used: entailment says *delivered*, contradiction says
*overpromises*, and a faithful-but-paraphrased headline lands in between instead of
being falsely accused. `POST /delivery_gap`, live in the web demo.

**3. In-browser inference.** Teacher → distilled MiniLM student → ONNX → dynamic INT8
(22.9 MB) running inside the MV3 extension via ONNX Runtime Web. Open devtools →
Network on any page: **zero requests**. A from-scratch JS WordPiece tokenizer is
verified byte-identical to the Python one (10/10 parity probes).

**4. Political neutrality by construction.** Political entities are masked *before*
tokenization in every model — Biden↔Trump swaps produce byte-identical scores, by
architecture rather than by hope. Enforced by a CheckList-style behavioral test
suite that gates model promotion in CI.

## Quickstart

```bash
make install          # venv + deps
make data             # ingest -> MinHash dedup -> leakage-safe splits
make train            # calibrated baselines + eval report
make serve            # FastAPI + web demo on :8000
```

Score a headline:

```bash
curl -s -X POST localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"headline": "You Won'\''t Believe What This Senator Said Next"}' | python3 -m json.tool
```

Check whether an article delivers its headline:

```bash
curl -s -X POST localhost:8000/delivery_gap \
  -H 'Content-Type: application/json' \
  -d '{"headline": "...", "body": "..."}'
```

Or open `http://localhost:8000/` for the interactive demo (score meters, highlighted
trigger phrases, delivery-gap checker).

## The browser extension

```bash
make data data-webis          # datasets
make train-transformer        # teacher (GPU recommended — see docs/GPU.md)
make distill export-onnx      # student -> INT8 ONNX
python apps/extension/build.py            # bundle model + tokenizer + ORT
python tools/test_tokenizer_parity.py     # must print 10/10
```

Then `chrome://extensions` → Developer mode → **Load unpacked** → `apps/extension/`.
Colored dots (● green/amber/red) appear next to headlines; the popup toggles between
private in-browser mode and the full-breakdown server API.

## LLM labeling without a paid key

The taxonomy labels can be upgraded from lexicon weak-labels to LLM labels — and you
don't need a paid API. Any of these work (same validation gate for all):

| Provider | Cost | Setup |
|---|---|---|
| `--provider gemini` | **free tier** | key from [aistudio.google.com](https://aistudio.google.com) → `GEMINI_API_KEY` |
| `--provider groq` | **free tier** | key from [console.groq.com](https://console.groq.com) → `GROQ_API_KEY` |
| `--provider openrouter` | free `:free` models | key from [openrouter.ai](https://openrouter.ai) → `OPENROUTER_API_KEY` |
| `--provider ollama` | free, fully local | [ollama.com](https://ollama.com), `ollama pull llama3.1:8b` |
| `--provider anthropic` | paid | `ANTHROPIC_API_KEY` |

```bash
python -m manipulens.labeling.llm_labeler validate-gold --provider gemini
python -m manipulens.labeling.llm_labeler label --provider gemini --limit 500
```

**The gate is provider-agnostic:** whichever model produces the labels, a dimension is
only used for training if its agreement with the human gold set clears α ≥ 0.6 —
weak labelers get filtered out automatically, not silently trusted.

## Engineering rigor

- **Leakage-safe evaluation** — MinHash near-dup dedup (213 caught in the benchmark
  corpus) + grouped splits. Random splits on headline data silently inflate every metric.
- **Calibrated scores** — isotonic calibration, ECE 0.0048. An uncalibrated user-facing
  probability is decoration.
- **Behavioral promotion gate** — entity swaps must not move scores; injected clickbait
  phrases must raise them. This suite caught a real bias (Democrats-vs-Republicans score
  delta) during development; the fix was masking by construction.
- **Self-healing training loop** — a preflight forward/backward on the real device plus
  per-batch loss/gradient-norm guards. When DeBERTa-v3 turned out to be numerically
  broken on Colab T4 GPUs (NaN gradients under transformers 5.x), the loop detected it
  in ~10 seconds, skipped nothing silently, and **fell back to RoBERTa automatically**.
- **Data contracts** — pandera schemas on every dataset boundary.
- **Honest labels** — the LLM-vs-gold validation report is published, including the
  dimensions that *failed*.

## Repo map

```
src/manipulens/
├── data/        # ingest (Chakraborty + Webis), MinHash dedup, splits, schemas
├── labeling/    # codebook.md, lexicons/, labeling functions, Krippendorff's α,
│                # LLM labeler (Anthropic/Gemini/Groq/OpenRouter/Ollama + gold gate)
├── models/      # baselines, calibration, multi-task transformer, distillation,
│                # ONNX INT8 export, delivery-gap NLI, entity masking
├── eval/        # metrics (ECE, PR-AUC)
└── api/         # FastAPI service + web demo
apps/extension/  # MV3 extension: in-browser ONNX inference, JS tokenizer
tools/           # icon generator, JS↔Python tokenizer parity check
tests/           # unit/, data/, behavioral/ (promotion gate)
docs/            # GPU runbook, ADRs, project plan
```

## Improving detection quality

The current model is trained on 2,459 graded-intensity posts. The single best upgrade
is the full Webis archive — **8× more intensity supervision**:

1. `params.yaml → webis.archive: clickbait17-train-170630.zip` (937 MB, 19,538 posts)
2. Re-run the GPU pipeline (`docs/GPU.md`) — data, train, distill, export
3. Rebuild the extension bundle

Second best: LLM taxonomy labels via a free provider (above). Third: bump
`transformer.epochs` to 3.

## CI

The workflow lives at `ci/github-workflows/ci.yml` (the deploy token here lacks the
`workflow` scope). To activate: `mkdir -p .github/workflows && cp ci/github-workflows/ci.yml .github/workflows/` and push with normal credentials. It runs lint, unit + data-contract
tests, then trains a sample model and runs the behavioral promotion gate.

## Limitations

ManipuLens estimates the presence of *rhetorical techniques*. It does **not** verify
factual accuracy; a high manipulation score does not mean a story is false, and the
delivery-gap verdict is a claim about headline↔body consistency, not about the world.
Training data is English-language news/social headlines from 2016–2017; expect degraded
performance on other domains (sports scores, non-English text). See
[ADR-0001](docs/decisions/0001-technique-detection-not-truth.md).
