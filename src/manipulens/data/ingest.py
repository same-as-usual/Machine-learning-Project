"""Ingest public benchmark headline data.

Primary source: Chakraborty et al. 2016 ("Stop Clickbait") — ~32k headlines,
binary clickbait labels, distributed via the authors' GitHub repo.

    clickbait_data.gz      — clickbait headlines (BuzzFeed, Upworthy, ...)
    non_clickbait_data.gz  — non-clickbait headlines (NYT, Guardian, ...)

`--sample` writes a small built-in synthetic sample instead (used by CI so the
pipeline is testable without network access).

Roadmap: add Webis Clickbait Challenge 2017 (graded intensity + article bodies)
as the anchor dataset for regression + delivery-gap training.
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys

import pandas as pd
import requests

from manipulens.config import data_dir, load_params
from manipulens.data.schemas import validate_raw

BASE_URL = "https://raw.githubusercontent.com/bhargaviparanjape/clickbait/master/dataset"
FILES = {
    1: f"{BASE_URL}/clickbait_data.gz",
    0: f"{BASE_URL}/non_clickbait_data.gz",
}

# Tiny synthetic sample for --sample mode (CI / offline smoke tests).
_SAMPLE = [
    ("You Won't Believe What This Senator Said Next", 1),
    ("27 INSANE Facts About Sleep That Will Blow Your Mind", 1),
    ("This One Trick Doctors Don't Want You To Know", 1),
    ("The Shocking Truth About Your Morning Coffee", 1),
    ("Here's What Happened When She Opened The Door", 1),
    ("10 Things Only 90s Kids Will Understand", 1),
    ("What This Dog Did Next Will Melt Your Heart", 1),
    ("The Terrifying Reason You Should Never Ignore This Symptom", 1),
    ("She Tried This Weird Hack And The Results Were Stunning", 1),
    ("Why Everyone Is Talking About This New Diet", 1),
    ("Senate passes infrastructure bill in 69-30 vote", 0),
    ("Federal Reserve holds interest rates steady", 0),
    ("Scientists report progress on malaria vaccine trial", 0),
    ("City council approves 2026 budget after public hearing", 0),
    ("Earnings: retailer posts 3% revenue growth in Q2", 0),
    ("Court upholds ruling in state redistricting case", 0),
    ("Storm system expected to bring rain to the Midwest this weekend", 0),
    ("Researchers find correlation between exercise and mood", 0),
    ("Governor signs education funding bill into law", 0),
    ("Study suggests link between diet and sleep quality", 0),
]


def _download(label: int, url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return pd.DataFrame(
        {
            "headline": lines,
            "label_clickbait": label,
            "source": "chakraborty2016",
            "outlet": "",
        }
    )


def build_sample(multiplier: int = 25) -> pd.DataFrame:
    """Synthetic sample: repeat the seed set with numbered suffix variants so
    dedup/splitting still has enough rows to exercise logic in CI."""
    rows = []
    for i in range(multiplier):
        for text, label in _SAMPLE:
            suffix = "" if i == 0 else f" — report {i}"
            rows.append(
                {
                    "headline": text + suffix,
                    "label_clickbait": label,
                    "source": "synthetic-sample",
                    "outlet": "",
                }
            )
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    params = load_params()["data"]
    df = df.copy()
    df["headline"] = df["headline"].str.strip()
    lengths = df["headline"].str.len()
    df = df[(lengths >= params["min_headline_chars"]) & (lengths <= params["max_headline_chars"])]
    df = df.drop_duplicates(subset="headline")  # exact dupes; near-dupes handled in dedup stage
    return df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="write built-in synthetic sample")
    args = parser.parse_args(argv)

    if args.sample:
        df = build_sample()
    else:
        parts = [_download(label, url) for label, url in FILES.items()]
        df = pd.concat(parts, ignore_index=True)

    df = clean(df)
    df = validate_raw(df)
    out = data_dir("raw") / "headlines.parquet"
    df.to_parquet(out, index=False)
    counts = df["label_clickbait"].value_counts().to_dict()
    print(f"wrote {len(df)} headlines -> {out} (label counts: {counts})")


if __name__ == "__main__":
    main(sys.argv[1:])
