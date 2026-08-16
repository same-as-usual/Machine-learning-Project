# ManipuLens — Headline Manipulation Analyzer (Upgraded, Professional Plan)

## Context

The user is building a portfolio project with genuine novelty: a model that scores news headlines for sensationalism/manipulation, served via API, web app, and browser extension. A basic plan existed (scrape → weak labels → LightGBM/DistilBERT → FastAPI → extension). The user asked to upgrade it to be "truly unbeatable and professional" — i.e., industry-level engineering rigor plus defensible novelty that stands out to reviewers who know ML.

**Strategic reframe (most important change):** drop the "lie detector" branding. A model cannot verify truth from a headline. Rebrand as detection of **manipulation techniques** — more defensible, more novel, and it turns the neutrality audit into a headline feature instead of an apology. Outputs say "uses curiosity-gap + outrage framing," never "this is a lie."

## The 4 Novelty Pillars

1. **Multi-dimensional manipulation taxonomy** (not binary clickbait): curiosity gap, outrage bait, fear-mongering, false certainty/overclaiming, emotional framing, sensational formatting — each on ordinal scales, backed by a written annotation codebook, gold-set inter-annotator agreement (Krippendorff's α), and a published dataset card.
2. **Headline↔body "delivery gap"**: cross-encoder NLI (DeBERTa-v3-NLI zero-shot first, fine-tuned on Webis Clickbait 2017 which ships article bodies) → does the headline overpromise vs. the article? The honest version of "lie detection."
3. **In-browser inference for privacy** (wow-factor): distill → MiniLM student → ONNX → INT8 (<30 MB) running inside the Chrome extension via ONNX Runtime Web. "Your browsing never leaves your machine" + demonstrates the full distillation/quantization/edge-deploy chain.
4. **Published political-neutrality audit**: score-distribution parity across outlet leanings (AllSides/Ad Fontes ratings, topic-matched), entity-counterfactual swap tests (Biden↔Trump etc. must not move scores), shipped as a standalone audit report + model card.

**Explicitly cut as solo-dev overkill:** cross-outlet framing comparison as a product feature (demoted to a notebook/blog-post analysis), fully automated retraining (replaced by one documented feedback→retrain cycle), Kubernetes/Airflow/feature stores/self-hosted MLflow/Great Expectations.

## Phased Plan (~12 weeks part-time; every phase independently shippable)

### Phase 0 — Professional skeleton (wk 1)
Monorepo scaffold, `pyproject.toml` (uv), ruff + mypy + pre-commit, pytest, GitHub Actions CI, DVC init + remote (B2/S3), taxonomy codebook v0, ADRs in `docs/decisions/`.
**Accept:** CI green; `dvc pull` from clean clone; codebook has definitions + 5 examples/dimension.

### Phase 1 — Data foundation (wk 2–3)
- Public benchmarks: **Webis Clickbait Challenge 2017** (~38k, graded intensity + bodies — anchor dataset), Chakraborty 2016 (~32k binary), optionally SemEval-2023 Task 5.
- RSS scraper, 20–30 outlets **balanced across political leanings**, headlines + metadata only (robots.txt respected; bodies come from clickbait17 → minimal legal exposure).
- MinHash near-dup dedup; **splits by time and outlet** (random splits leak via near-dupes and inflate all metrics — the #1 mistake in this domain); pandera schemas as data contracts in CI; DVC pipeline stage; dataset card.
**Accept:** ≥50k deduped, versioned headlines; schema tests in CI; documented split protocol.

### Phase 2 — Labeling with rigor (wk 3–4)
- Codebook v1 after pilot round.
- **Gold set 600–800 headlines, double-annotated**; report Krippendorff's α per dimension; **merge/drop any dimension below α ≈ 0.6** (pre-committed rule — pruning your own taxonomy is a portfolio talking point).
- **LLM-as-labeler, validated** against the gold set per dimension before scaling; prompt ensemble + self-consistency; cache all labels + rationales; hard budget cap.
- Weak-supervision labeling functions (outrage/fear lexicons; regexes: "You Won't Believe", forward references, listicle patterns, ALL-CAPS ratio) via a simple label model — these double as explanation features and CheckList probes later.
**Accept:** 30–50k labeled set in DVC; written label-validation report (α + LLM-vs-gold tables).

### Phase 3 — Modeling ladder + eval harness (wk 5–6)
- Ladder kept forever in the eval table: lexicon rules → TF-IDF + LogReg → LightGBM → **DeBERTa-v3-small multi-task head** (clickbait17 intensity regression + taxonomy multi-label).
- **Calibration** via temperature scaling; report ECE + reliability diagrams (a user-facing score must be calibrated).
- **CheckList-style behavioral suite** in `tests/behavioral/`, run in CI as a *model promotion gate*: invariance (entity/synonym swaps must not flip scores), directional expectations (prepending clickbait phrases must raise curiosity-gap), robustness (typos, casing).
- W&B experiment tracking; every README table traceable to a run.
**Accept:** transformer beats TF-IDF by stated margin (Spearman vs. clickbait17 intensity; macro-F1 taxonomy); near published Webis 2017 leaderboard numbers (cited); ECE < 0.05; behavioral pass thresholds met.

### Phase 4 — Compression for the edge (wk 7)
Distill to MiniLM student on soft labels → ONNX → INT8 dynamic quantization → benchmark harness.
**Accept:** ≤2–3 pt quality drop; <30 MB; p95 <60 ms CPU/headline; benchmark table committed.

### Phase 5 — Serving + delivery-gap + web demo (wk 8–9)
- FastAPI (Pydantic v2): `POST /score` (batch → per-dimension calibrated scores + trigger-phrase spans), `POST /delivery-gap` (headline+body NLI), `POST /feedback`. ONNX Runtime inference — **no torch in prod image**.
- Redis cache (headline-hash keys — headlines repeat massively), slowapi rate limiting, structlog, Prometheus + Grafana, docker-compose, Locust load test, deploy to Fly.io/Hetzner.
- Web demo (Vite+React): paste headline/URL → radar chart of dimensions + highlighted trigger phrases + delivery-gap verdict.
**Accept:** p95 <100 ms warm; 100 RPS locally; public demo URL; Grafana dashboard.

### Phase 6 — Extension + feedback loop (wk 10–11)
MV3 extension badges scores on news-site headlines; **privacy toggle**: server mode vs. fully in-browser ONNX Runtime Web (zero network calls, verifiable in devtools); "I disagree" → feedback endpoint → SQLite/Postgres queue → weekly DVC export; **one demonstrated review→relabel→retrain→redeploy cycle** written as a runbook.
Watch out: homepage link text ≠ article headlines — build a small link-text eval slice + augmentation before sign-off.
**Accept:** works on 5 major news sites; in-browser mode makes zero network calls; one feedback-driven retrain documented.

### Phase 7 — Audits, drift, packaging (wk 12)
- **Neutrality audit report** (`reports/neutrality_audit.md`): leaning parity + entity counterfactuals.
- Robustness report: LLM-paraphrase attack success rate.
- Drift: Evidently batch job on GitHub Actions cron over logged scores vs. training distribution → report + Grafana panel.
- Model card + dataset card; **publish dataset + ONNX model on Hugging Face Hub** (public model registry + distribution + visibility).
- Cross-outlet framing comparison notebook/blog post; README with architecture diagram + demo GIF.
**Accept:** all reports committed; HF artifacts public; a stranger runs `docker compose up` and scores a headline in <5 min.

## Tech Stack (opinionated)

| Concern | Choice | Why |
|---|---|---|
| Env | uv + pyproject | modern, fast |
| Data versioning | DVC + B2/S3; final snapshot → HF Hub | pipelines + versioning in one |
| Data validation | pandera | CI-friendly; GE is enterprise ceremony |
| Tracking | W&B free tier | zero-ops; shareable reports are portfolio artifacts |
| Registry | HF Hub | public + doubles as distribution |
| Models | TF-IDF/LogReg → LightGBM → DeBERTa-v3-small multi-task → MiniLM student | defensible ladder |
| NLI | DeBERTa-v3-NLI cross-encoder | cheapest path to delivery-gap |
| Compression | ONNX + INT8 | one artifact serves API, extension, browser |
| Serving | FastAPI + ONNX Runtime + Redis + slowapi + structlog | torch-free prod image |
| Monitoring | Prometheus/Grafana; Evidently on cron | real observability, no platform team |
| CI/CD | GitHub Actions incl. behavioral-test promotion gate | the standout detail |
| Frontend | Vite+React demo; MV3 extension + onnxruntime-web | |
| Deploy | Fly.io or Hetzner VPS | cheap, honest |

## Top Risks & Mitigations
1. "Lie detector" credibility/defamation → technique-detection reframe; calibrated per-technique scores; explicit limitations in model card.
2. Subjective dimensions (low IAA) → pilot + tight codebook + pre-committed α ≈ 0.6 merge/drop rule.
3. Political bias → leaning-balanced scraping; outlet identity never a feature; counterfactual tests in CI, not just the final audit; publish results even if imperfect.
4. LLM-label circularity/cost → gold-set validation first; caching; budget cap; LLM-independent weak-supervision signal.
5. Split leakage → MinHash dedup + time/outlet splits enforced by CI data test.
6. Scraping legality → RSS headlines only; bodies from clickbait17 or the user's own open page.
7. Extension domain shift → link-text eval slice + augmentation.
8. Scope creep → per-phase acceptance criteria; project is complete and rigorous even if stopped after Phase 5.

## Repo Structure

```
manipulens/
├── pyproject.toml  .pre-commit-config.yaml  Makefile
├── dvc.yaml  params.yaml
├── configs/
├── src/manipulens/
│   ├── data/        # ingest.py, scrape_rss.py, dedup.py, splits.py, schemas.py
│   ├── labeling/    # codebook.md, lexicons/, labeling_functions.py, llm_labeler.py, agreement.py
│   ├── models/      # baselines.py, transformer.py, distill.py, calibrate.py, export_onnx.py
│   ├── eval/        # metrics.py, behavioral/, robustness.py, neutrality_audit.py
│   └── api/         # main.py, inference.py, cache.py, feedback.py
├── tests/           # unit/, data/, behavioral/ (CI promotion gate)
├── apps/web/  apps/extension/
├── docker/  docker-compose.yml
├── .github/workflows/   # ci.yml, train-gate.yml, drift-cron.yml, deploy.yml
├── notebooks/  reports/  docs/decisions/
```

## Critical files (first implementation targets)
- `src/manipulens/labeling/codebook.md` — taxonomy definitions everything depends on
- `dvc.yaml` — reproducible data→label→train→export spine
- `src/manipulens/data/splits.py` — leakage-safe dedup + splitting (make-or-break for every metric)
- `src/manipulens/models/transformer.py` — core multi-task encoder
- `src/manipulens/api/main.py` — serving surface for web app + extension

## Verification (end-to-end)
- **CI**: lint/type/unit/data/behavioral tests green on every PR; behavioral suite gates model promotion.
- **Data**: `dvc repro` from clean clone rebuilds dataset; split-leakage test passes.
- **Model**: eval table reproduces from W&B runs; ECE < 0.05; behavioral pass thresholds; benchmark vs. Webis 2017 leaderboard.
- **Serving**: `docker compose up` → `curl POST /score` returns calibrated per-dimension scores <100 ms warm; Locust at 100 RPS; Grafana shows metrics.
- **Extension**: load unpacked on chrome, visit 5 news sites, badges appear; devtools network tab shows zero calls in privacy mode.
- **Full loop**: submit feedback via extension → appears in queue → run documented retrain runbook once.

## Execution order for the first working slice (MVP-first)
Phase 0 scaffold → clickbait17 ingest + dedup/splits → weak-supervision + TF-IDF baseline → FastAPI `/score` with the baseline → then climb the ladder. Commit at every milestone.
