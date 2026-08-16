# Gold set

`gold_seed.csv` is a **bootstrap seed**: 30 headlines annotated once by the
project author following `codebook.md` v0.1. It exists so the LLM-validation
pipeline (`manipulens.labeling.llm_labeler validate-gold`) is runnable from day
one.

**It is not yet a real gold set.** The reliability policy (codebook.md) requires
600–800 headlines, double-annotated, with per-dimension Krippendorff's alpha
reported — and dimensions below α ≈ 0.6 merged or dropped. Until that
annotation round happens, treat validation numbers against this seed as
plumbing checks, not evidence.

Columns: `headline` + one 0–2 ordinal score per taxonomy dimension.
