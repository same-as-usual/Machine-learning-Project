"""Ingest the Webis Clickbait Challenge 2017 corpus (Zenodo record 5530410).

This is the anchor dataset for two pillars:
  1. graded clickbait *intensity* (truthMean in [0,1], 5 crowd judgments/post)
     -> regression target for the transformer
  2. article bodies (targetTitle + targetParagraphs)
     -> headline<->body "delivery gap" NLI training/eval

Default file: clickbait17-train-170331.zip (147.8 MB, 2,459 posts) — small
enough for constrained environments. Switch to clickbait17-train-170630.zip
(937 MB, 19,538 posts) via params.yaml when disk allows.

`--sample` writes a synthetic sample (CI / offline).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

from manipulens.config import data_dir, load_params

ZENODO_URL = "https://zenodo.org/records/5530410/files/{name}?download=1"

_SAMPLE_ROWS = [
    # (teaser post, target title, body, truthMean)
    (
        "You won't believe what this senator said next",
        "Senator's remarks on budget draw attention",
        "The senator spoke for twenty minutes about routine budget appropriations. "
        "The remarks covered standard procedural matters and drew polite applause.",
        0.93,
    ),
    (
        "Senate passes infrastructure bill",
        "Senate passes $1.2T infrastructure bill 69-30",
        "The Senate approved the bipartisan infrastructure package on Tuesday, "
        "sending the measure to the House. The bill funds roads, bridges and broadband.",
        0.03,
    ),
    (
        "This one trick will change how you cook forever",
        "A guide to sharpening kitchen knives",
        "Regular honing keeps a knife edge aligned. Professional sharpeners "
        "recommend a whetstone at a consistent angle for best results.",
        0.87,
    ),
    (
        "Fed holds interest rates steady",
        "Federal Reserve holds interest rates steady",
        "The Federal Reserve kept its benchmark rate unchanged, citing stable "
        "inflation expectations and a balanced labor market.",
        0.0,
    ),
    (
        "The terrifying reason you should never ignore this symptom",
        "Doctors discuss when fatigue warrants a checkup",
        "Persistent fatigue can have many causes, most of them benign. "
        "Physicians suggest a routine appointment if it lasts several weeks.",
        0.97,
    ),
    (
        "Study finds moderate exercise improves mood",
        "Study suggests link between exercise and mood",
        "Researchers followed 2,000 adults for a year and found a modest "
        "correlation between regular moderate exercise and self-reported mood.",
        0.1,
    ),
] * 40  # enough rows to exercise splits in CI


def build_sample() -> pd.DataFrame:
    rows = []
    for i, (post, title, body, mean) in enumerate(_SAMPLE_ROWS):
        suffix = "" if i < 6 else f" — report {i}"
        rows.append(
            {
                "id": f"sample-{i}",
                "headline": post + suffix,
                "target_title": title,
                "body": body,
                "intensity": mean,
                "label_clickbait": int(mean >= 0.5),
                "source": "webis-sample",
            }
        )
    return pd.DataFrame(rows)


def _read_jsonl_from_zip(zf: zipfile.ZipFile, suffix: str) -> list[dict]:
    name = next(n for n in zf.namelist() if n.endswith(suffix))
    with zf.open(name) as f:
        return [json.loads(line) for line in io.TextIOWrapper(f, encoding="utf-8") if line.strip()]


def parse_zip(zip_path: Path, max_body_chars: int) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        instances = _read_jsonl_from_zip(zf, "instances.jsonl")
        truth = {t["id"]: t for t in _read_jsonl_from_zip(zf, "truth.jsonl")}

    rows = []
    for inst in instances:
        t = truth.get(inst["id"])
        if t is None:
            continue
        post_text = " ".join(inst.get("postText", [])).strip()
        body = " ".join(p.strip() for p in inst.get("targetParagraphs", []) if p.strip())
        rows.append(
            {
                "id": str(inst["id"]),
                "headline": post_text,
                "target_title": (inst.get("targetTitle") or "").strip(),
                "body": body[:max_body_chars],
                "intensity": float(t["truthMean"]),
                "label_clickbait": int(t["truthClass"] == "clickbait"),
                "source": "webis-clickbait17",
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["headline"].str.len() >= 5].reset_index(drop=True)
    return df


def download(name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"using cached {dest}")
        return dest
    url = ZENODO_URL.format(name=name)
    print(f"downloading {url} ...")
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="write built-in synthetic sample")
    parser.add_argument("--keep-zip", action="store_true", help="keep the downloaded archive")
    args = parser.parse_args(argv)

    params = load_params()["webis"]
    if args.sample:
        df = build_sample()
    else:
        zip_path = data_dir("raw") / params["archive"]
        download(params["archive"], zip_path)
        df = parse_zip(zip_path, max_body_chars=params["max_body_chars"])
        if not args.keep_zip:
            zip_path.unlink()  # disk-constrained environments

    out = data_dir("raw") / "webis.parquet"
    df.to_parquet(out, index=False)
    print(
        f"wrote {len(df)} posts -> {out} "
        f"(mean intensity {df['intensity'].mean():.3f}, "
        f"clickbait share {df['label_clickbait'].mean():.2%}, "
        f"with-body share {(df['body'].str.len() > 0).mean():.2%})"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
