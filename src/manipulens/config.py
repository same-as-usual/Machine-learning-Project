"""Central config: loads params.yaml from the repo root."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = REPO_ROOT / "params.yaml"


@functools.lru_cache(maxsize=1)
def load_params() -> dict[str, Any]:
    with PARAMS_FILE.open() as f:
        return yaml.safe_load(f)


def data_dir(kind: str) -> Path:
    """Resolve a data directory ('raw_dir' | 'interim_dir' | 'processed_dir')."""
    p = REPO_ROOT / load_params()["data"][f"{kind}_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def artifacts_dir(kind: str = "models") -> Path:
    key = "models_dir" if kind == "models" else "reports_dir"
    p = REPO_ROOT / load_params()["artifacts"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
