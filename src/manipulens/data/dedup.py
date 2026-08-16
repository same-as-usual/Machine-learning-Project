"""MinHash-LSH near-duplicate detection.

Headline corpora are riddled with near-duplicates (wire copy, syndication,
A/B-tested variants). A random split leaks these across train/test and inflates
every reported metric. We cluster near-dupes here and split BY CLUSTER later.

Each row gets a `dup_group` id; near-duplicates share a group. Groups are the
atomic unit for splitting (see splits.py).
"""

from __future__ import annotations

import re
import sys

import pandas as pd
from datasketch import MinHash, MinHashLSH

from manipulens.config import data_dir, load_params
from manipulens.data.schemas import validate_deduped, validate_raw

_norm_re = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    return _norm_re.sub("", text.lower()).strip()


def char_shingles(text: str, k: int) -> set[bytes]:
    text = normalize(text)
    if len(text) <= k:
        return {text.encode()}
    return {text[i : i + k].encode() for i in range(len(text) - k + 1)}


def assign_dup_groups(headlines: list[str], num_perm: int, threshold: float, k: int) -> list[int]:
    """Union near-duplicate pairs found via LSH into connected components."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: list[MinHash] = []
    parent = list(range(len(headlines)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, text in enumerate(headlines):
        m = MinHash(num_perm=num_perm)
        for sh in char_shingles(text, k):
            m.update(sh)
        # query BEFORE insert: link to all existing near-dupes
        for j in lsh.query(m):
            union(i, int(j))
        lsh.insert(str(i), m)
        minhashes.append(m)

    # compress to consecutive group ids
    roots = [find(i) for i in range(len(headlines))]
    remap: dict[int, int] = {}
    return [remap.setdefault(r, len(remap)) for r in roots]


def main(argv: list[str] | None = None) -> None:
    params = load_params()["dedup"]
    df = pd.read_parquet(data_dir("raw") / "headlines.parquet")
    df = validate_raw(df)

    groups = assign_dup_groups(
        df["headline"].tolist(),
        num_perm=params["num_perm"],
        threshold=params["lsh_threshold"],
        k=params["shingle_size"],
    )
    df = df.assign(dup_group=groups)

    n_groups = df["dup_group"].nunique()
    n_dupes = len(df) - n_groups
    df = validate_deduped(df)
    out = data_dir("interim") / "deduped.parquet"
    df.to_parquet(out, index=False)
    print(
        f"{len(df)} rows -> {n_groups} near-dup groups ({n_dupes} rows have a near-dupe) -> {out}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
