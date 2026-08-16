"""LLM-as-labeler for the manipulation taxonomy, validated against a human gold set.

Design (per the label-rigor plan):
  - The full annotation codebook is the system prompt, with prompt caching
    (cache_control on the codebook block) so repeated calls reuse the prefix.
  - Structured output via output_config.format (json_schema) — the API
    guarantees parseable JSON; no regex extraction.
  - Self-consistency: N independent samples per headline, per-dimension median.
  - Hard budget caps: max headlines per run AND max USD per run (estimated
    from usage on the fly); the run stops when either is hit.
  - Everything is cached to a JSONL file keyed by (headline, model, codebook
    version) — a re-run never re-bills.
  - VALIDATION GATE: before labels are used for training, per-dimension
    Krippendorff's alpha vs the human gold set must clear min_gold_alpha.
    Dimensions below the bar are excluded (reported, not hidden).

Usage:
  python -m manipulens.labeling.llm_labeler validate-gold   # LLM vs gold agreement
  python -m manipulens.labeling.llm_labeler label --limit 200
  python -m manipulens.labeling.llm_labeler label --dry-run  # no API, LF-simulated

Requires ANTHROPIC_API_KEY (or `ant auth login` profile) unless --dry-run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from importlib import resources
from pathlib import Path

import pandas as pd

from manipulens.config import REPO_ROOT, data_dir, load_params
from manipulens.labeling.agreement import agreement_report
from manipulens.labeling.labeling_functions import DIMENSIONS, score_headline

CACHE_FILE = REPO_ROOT / "data" / "labels" / "llm_labels.jsonl"
GOLD_FILE = REPO_ROOT / "data" / "gold" / "gold_seed.csv"
REPORT_FILE = REPO_ROOT / "reports" / "llm_label_validation.json"
CODEBOOK_VERSION = "0.1"

LABEL_SCHEMA = {
    "type": "object",
    "properties": {dim: {"type": "integer", "enum": [0, 1, 2]} for dim in DIMENSIONS}
    | {"out_of_scope": {"type": "boolean"}},
    "required": [*DIMENSIONS, "out_of_scope"],
    "additionalProperties": False,
}


def codebook_text() -> str:
    return (resources.files("manipulens.labeling") / "codebook.md").read_text()


def build_system() -> list[dict]:
    """Codebook as a cached system block: stable prefix first, cache_control on
    the last stable block so repeated labeling calls reuse it."""
    return [
        {
            "type": "text",
            "text": (
                "You are an expert annotator for the ManipuLens project. Apply the "
                "annotation codebook below EXACTLY as written. Score each dimension "
                "independently on its 0-2 ordinal scale. When torn between two "
                "scores, take the lower one (conservative default). Do not judge "
                "truthfulness — only rhetorical technique.\n\n" + codebook_text()
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _load_cache() -> dict[str, dict]:
    if not CACHE_FILE.exists():
        return {}
    out: dict[str, dict] = {}
    with CACHE_FILE.open() as f:
        for line in f:
            rec = json.loads(line)
            out[rec["key"]] = rec
    return out


def _cache_key(headline: str, model: str) -> str:
    return f"{model}|cb{CODEBOOK_VERSION}|{headline}"


def _append_cache(rec: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# labeling
# ---------------------------------------------------------------------------


def _label_once_api(client, model: str, system: list[dict], headline: str) -> tuple[dict, float]:
    """One structured-output labeling call. Returns (labels, usd_cost)."""
    params = load_params()["llm_labeling"]
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=system,
        output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
        messages=[{"role": "user", "content": f"Annotate this headline:\n\n{headline}"}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    labels = json.loads(text)
    u = response.usage
    cached_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    usd = (
        (u.input_tokens + 1.25 * cache_write + 0.1 * cached_read) * params["input_usd_per_mtok"]
        + u.output_tokens * params["output_usd_per_mtok"]
    ) / 1e6
    return labels, usd


def _label_once_dry(headline: str) -> tuple[dict, float]:
    """Dry-run: simulate labels from labeling functions (0/0.6/1.0 -> 0/1/2)."""
    results = score_headline(headline)
    labels = {
        dim: (0 if res.score < 0.3 else 1 if res.score < 0.8 else 2) for dim, res in results.items()
    }
    labels["out_of_scope"] = False
    return labels, 0.0


def label_headlines(
    headlines: list[str], dry_run: bool = False, limit: int | None = None
) -> pd.DataFrame:
    """Label with self-consistency + caching + budget caps. Returns a DataFrame
    with per-dimension median scores in {0,1,2}."""
    params = load_params()["llm_labeling"]
    model = params["model"]
    n_samples = params["n_samples"]
    max_headlines = min(limit or params["max_headlines"], params["max_headlines"])
    budget = params["max_usd_per_run"]

    cache = _load_cache()
    client = None
    system = None
    if not dry_run:
        import anthropic

        client = anthropic.Anthropic()
        system = build_system()

    spent = 0.0
    rows = []
    for headline in headlines[:max_headlines]:
        key = _cache_key(headline, model if not dry_run else "dry-run")
        if key in cache:
            rows.append({"headline": headline, **cache[key]["labels"]})
            continue
        if not dry_run and spent >= budget:
            print(f"budget cap ${budget:.2f} reached after {len(rows)} headlines; stopping")
            break

        samples = []
        for _ in range(n_samples if not dry_run else 1):
            if dry_run:
                labels, usd = _label_once_dry(headline)
            else:
                labels, usd = _label_once_api(client, model, system, headline)
            samples.append(labels)
            spent += usd

        # self-consistency: per-dimension median across samples
        agg = {dim: int(statistics.median(s[dim] for s in samples)) for dim in DIMENSIONS}
        agg["out_of_scope"] = sum(s["out_of_scope"] for s in samples) > len(samples) / 2
        _append_cache({"key": key, "labels": agg, "n_samples": len(samples)})
        cache[key] = {"labels": agg}
        rows.append({"headline": headline, **agg})

    if not dry_run:
        print(f"spent ~${spent:.3f} this run ({len(rows)} headlines)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# gold-set validation gate
# ---------------------------------------------------------------------------


def validate_against_gold(dry_run: bool = False) -> dict:
    """Label the gold set with the LLM and compute per-dimension agreement.
    Dimensions with alpha < min_gold_alpha are flagged as NOT usable."""
    params = load_params()["llm_labeling"]
    gold = pd.read_csv(GOLD_FILE)
    llm = label_headlines(gold["headline"].tolist(), dry_run=dry_run)
    merged = gold.merge(llm, on="headline", suffixes=("_gold", "_llm"))

    report: dict = {"n": len(merged), "min_gold_alpha": params["min_gold_alpha"], "dimensions": {}}
    for dim in DIMENSIONS:
        rep = agreement_report(
            merged[f"{dim}_gold"].tolist(), merged[f"{dim}_llm"].tolist(), metric="interval"
        )
        rep["usable"] = rep["alpha"] >= params["min_gold_alpha"]
        report["dimensions"][dim] = rep

    report["usable_dimensions"] = [d for d, r in report["dimensions"].items() if r["usable"]]
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"report -> {REPORT_FILE}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_label = sub.add_parser("label", help="label training headlines")
    p_label.add_argument("--limit", type=int, default=None)
    p_label.add_argument("--dry-run", action="store_true")
    p_val = sub.add_parser("validate-gold", help="LLM vs human gold agreement")
    p_val.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.cmd == "validate-gold":
        validate_against_gold(dry_run=args.dry_run)
    else:
        train_path = data_dir("processed") / "train.parquet"
        headlines = pd.read_parquet(train_path)["headline"].tolist()
        df = label_headlines(headlines, dry_run=args.dry_run, limit=args.limit)
        out = Path(REPO_ROOT / "data" / "labels" / "taxonomy_labels.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"wrote {len(df)} labeled headlines -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
