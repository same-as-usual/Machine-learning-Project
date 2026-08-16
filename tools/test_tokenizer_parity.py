#!/usr/bin/env python3
"""Verify the extension's JS preprocessing matches Python training-time
preprocessing exactly: maskEntities + WordPiece token ids must be identical.

Run after apps/extension/build.py:  .venv/bin/python tools/test_tokenizer_parity.py
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "apps" / "extension"
MAX_LENGTH = 64

PROBES = [
    "You Won't Believe What This Senator Said Next",
    "Senate passes $1.2T infrastructure bill 69-30",
    "27 INSANE budget facts!! #9 will blow your mind",
    "Biden responds to questions about the budget",
    "Trump slams critics in fiery speech",
    "Democrats and Republicans clash over the debt ceiling",
    "The silent killer hiding in your kitchen — doctors sound alarm",
    "Café reopens after rénovation, crowds celebrate",  # accents
    "Fed holds rates steady; markets shrug",
    "This one trick doctors don't want you to know",
]

NODE_SCRIPT = r"""
// NB: with `node -e`, user args start at process.argv[1]
const EXT = process.argv[1];
const MAXLEN = parseInt(process.argv[2]);
const fs = require("fs");
const { WordPieceTokenizer } = require(EXT + "/lib/tokenizer.js");
require(EXT + "/assets/political_entities.js"); // sets globalThis.MANIPULENS_POLITICAL_ENTITIES
const { maskEntities } = require(EXT + "/lib/mask.js");

const vocab = fs.readFileSync(EXT + "/assets/vocab.txt", "utf8");
const tok = new WordPieceTokenizer(vocab);
const probes = JSON.parse(fs.readFileSync(0, "utf8"));
const out = probes.map((p) => {
  const masked = maskEntities(p);
  return { masked, ids: tok.encode(masked, MAXLEN).inputIds };
});
process.stdout.write(JSON.stringify(out));
"""


def main() -> int:
    from transformers import AutoTokenizer

    from manipulens.models.neutralize import mask_entities

    js = json.loads(
        subprocess.run(
            ["node", "-e", NODE_SCRIPT, str(EXT), str(MAX_LENGTH)],
            input=json.dumps(PROBES),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )

    tokenizer = AutoTokenizer.from_pretrained(REPO / "models" / "artifacts" / "student")
    failures = 0
    for probe, js_result in zip(PROBES, js, strict=True):
        py_masked = mask_entities(probe)
        enc = tokenizer(py_masked, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        py_ids = enc["input_ids"]
        mask_ok = py_masked == js_result["masked"]
        ids_ok = py_ids == js_result["ids"]
        status = "OK  " if (mask_ok and ids_ok) else "FAIL"
        print(f"{status} {probe[:60]}")
        if not mask_ok:
            failures += 1
            print(f"     mask py: {py_masked!r}\n     mask js: {js_result['masked']!r}")
        if not ids_ok:
            failures += 1
            for i, (a, b) in enumerate(zip(py_ids, js_result["ids"], strict=True)):
                if a != b:
                    print(f"     first id diff at {i}: py={a} js={b}")
                    break
    print(f"\n{len(PROBES) - failures}/{len(PROBES)} probes identical")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
