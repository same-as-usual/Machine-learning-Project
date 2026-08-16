"""Multi-task transformer: shared encoder + three heads.

  intensity : regression on Webis clickbait17 truthMean (graded 0..1)
  taxonomy  : 6 technique dimensions (soft targets from labeling functions,
              upgraded to LLM/human labels when available)
  binary    : clickbait yes/no on the Chakraborty corpus

Political entities are masked before tokenization (neutrality by construction,
same policy as the baseline). Plain torch loop — no Trainer/accelerate
dependency, easy to read, runs on CPU for smoke tests.

Usage:
  python -m manipulens.models.transformer                     # full fine-tune
  python -m manipulens.models.transformer --smoke             # tiny model, few steps
  python -m manipulens.models.transformer --model-name X ...  # overrides

Writes: models/artifacts/transformer/ (weights, tokenizer, heads, config)
        reports/transformer_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from manipulens.config import REPO_ROOT, artifacts_dir, data_dir, load_params
from manipulens.labeling.labeling_functions import DIMENSIONS, score_headline
from manipulens.models.neutralize import mask_entities

TASKS = ("intensity", "taxonomy", "binary")


def cap_torch_threads() -> None:
    """CPU thread oversubscription makes torch ~25x slower on shared/throttled
    cores (measured: 45s vs 1.8s per MiniLM forward at 48 vs 8 threads).
    Cap to a sane default; override with MANIPULENS_TORCH_THREADS."""
    import os

    if not torch.cuda.is_available():
        n = int(os.environ.get("MANIPULENS_TORCH_THREADS", "8"))
        torch.set_num_threads(n)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


LLM_LABELS_FILE = REPO_ROOT / "data" / "labels" / "taxonomy_labels.parquet"
LLM_VALIDATION_FILE = REPO_ROOT / "reports" / "llm_label_validation.json"


def load_llm_taxonomy(
    labels_file: Path | None = None, validation_file: Path | None = None
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Validated LLM taxonomy labels, gated by the gold-set agreement report.

    Returns (headline -> {dimension: target in [0,1]}, usable_dimensions).
    Only dimensions that cleared min_gold_alpha in the validation report are
    included — the gate is enforced HERE, at the training boundary, not just
    reported. Labels use the codebook's 0-2 scale, mapped to 0/0.5/1 targets.
    """
    labels_file = labels_file or LLM_LABELS_FILE
    validation_file = validation_file or LLM_VALIDATION_FILE
    if not labels_file.exists() or not validation_file.exists():
        return {}, []
    usable = json.loads(validation_file.read_text()).get("usable_dimensions", [])
    if not usable:
        return {}, []
    df = pd.read_parquet(labels_file)
    if "out_of_scope" in df.columns:
        df = df[~df["out_of_scope"].astype(bool)]
    mapping = {
        row["headline"]: {dim: float(row[dim]) / 2.0 for dim in usable} for _, row in df.iterrows()
    }
    return mapping, usable


def taxonomy_targets(
    headline: str,
    llm_map: dict[str, dict[str, float]] | None = None,
    usable: list[str] | None = None,
) -> list[float]:
    """Per-dimension targets: labeling-function weak labels, overridden by
    validated LLM labels where available (usable dimensions only)."""
    results = score_headline(headline)
    targets = [results[d].score for d in DIMENSIONS]
    if llm_map and (llm_labels := llm_map.get(headline)):
        for i, dim in enumerate(DIMENSIONS):
            if usable and dim in usable and dim in llm_labels:
                targets[i] = llm_labels[dim]
    return targets


def build_frame(smoke: bool = False) -> pd.DataFrame:
    """Combine Webis (intensity) + Chakraborty (binary) into one frame.

    Columns: headline, intensity (nan if absent), label (nan if absent),
    taxonomy targets are computed on the fly (weak labels).
    """
    frames = []
    webis_path = data_dir("raw") / "webis.parquet"
    if webis_path.exists():
        w = pd.read_parquet(webis_path)
        frames.append(
            pd.DataFrame(
                {
                    "headline": w["headline"],
                    "intensity": w["intensity"],
                    "label": w["label_clickbait"].astype(float),
                }
            )
        )
    chak_path = data_dir("processed") / "train.parquet"
    if chak_path.exists():
        c = pd.read_parquet(chak_path)
        frames.append(
            pd.DataFrame(
                {
                    "headline": c["headline"],
                    "intensity": np.nan,
                    "label": c["label_clickbait"].astype(float),
                }
            )
        )
    if not frames:
        raise FileNotFoundError("no training data; run ingest + ingest_webis first")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["headline"])
    if smoke:
        df = df.sample(n=min(256, len(df)), random_state=0).reset_index(drop=True)
    return df


class HeadlineDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        llm_map: dict[str, dict[str, float]] | None = None,
        usable: list[str] | None = None,
    ):
        self.headlines = [mask_entities(h) for h in df["headline"].tolist()]
        self.intensity = df["intensity"].to_numpy(dtype=np.float32)
        self.label = df["label"].to_numpy(dtype=np.float32)
        self.taxonomy = np.array(
            [taxonomy_targets(h, llm_map, usable) for h in df["headline"]], dtype=np.float32
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.headlines)

    def __getitem__(self, i: int) -> dict:
        enc = self.tokenizer(
            self.headlines[i],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "intensity": torch.tensor(self.intensity[i]),
            "label": torch.tensor(self.label[i]),
            "taxonomy": torch.tensor(self.taxonomy[i]),
        }


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


class MultiTaskHeadlineModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.intensity_head = nn.Linear(hidden, 1)
        self.taxonomy_head = nn.Linear(hidden, len(DIMENSIONS))
        self.binary_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # mean pooling over non-pad tokens (robust across encoder families)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        return {
            "intensity": self.intensity_head(pooled).squeeze(-1),
            "taxonomy": self.taxonomy_head(pooled),
            "binary": self.binary_head(pooled).squeeze(-1),
        }


def multitask_loss(outputs: dict, batch: dict, weights: dict[str, float]) -> torch.Tensor:
    """Masked losses: each task only contributes where its target exists."""
    total = torch.tensor(0.0, device=outputs["intensity"].device)

    m = ~torch.isnan(batch["intensity"])
    if m.any():
        pred = torch.sigmoid(outputs["intensity"][m])
        total = total + weights["intensity"] * nn.functional.mse_loss(pred, batch["intensity"][m])

    total = total + weights["taxonomy"] * nn.functional.binary_cross_entropy_with_logits(
        outputs["taxonomy"], batch["taxonomy"]
    )

    m = ~torch.isnan(batch["label"])
    if m.any():
        total = total + weights["binary"] * nn.functional.binary_cross_entropy_with_logits(
            outputs["binary"][m], batch["label"][m]
        )
    return total


# --------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    intensity_pred, intensity_true = [], []
    binary_pred, binary_true = [], []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        m = ~torch.isnan(batch["intensity"])
        if m.any():
            intensity_pred += torch.sigmoid(out["intensity"].cpu()[m]).tolist()
            intensity_true += batch["intensity"][m].tolist()
        m = ~torch.isnan(batch["label"])
        if m.any():
            binary_pred += torch.sigmoid(out["binary"].cpu()[m]).tolist()
            binary_true += batch["label"][m].tolist()

    report: dict[str, float] = {}
    if len(intensity_true) >= 8:
        rho = spearmanr(intensity_true, intensity_pred).statistic
        report["intensity_spearman"] = float(rho)
        report["intensity_mae"] = float(
            np.mean(np.abs(np.array(intensity_true) - np.array(intensity_pred)))
        )
    if len(binary_true) >= 8 and len(set(binary_true)) > 1:
        from sklearn.metrics import roc_auc_score

        report["binary_roc_auc"] = float(roc_auc_score(binary_true, binary_pred))
    report["n_eval"] = float(len(intensity_true) + len(binary_true))
    return report


def train(args: argparse.Namespace) -> dict:
    cap_torch_threads()
    params = load_params()["transformer"]
    model_name = args.model_name or params["model_name"]
    torch.manual_seed(params["seed"])

    df = build_frame(smoke=args.smoke)
    val_frac = 0.15
    val = df.sample(frac=val_frac, random_state=params["seed"])
    trn = df.drop(val.index).reset_index(drop=True)
    val = val.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = params["max_length"]
    llm_map, usable = load_llm_taxonomy()
    if llm_map:
        n_hit = sum(1 for h in df["headline"] if h in llm_map)
        print(
            f"validated LLM labels: {len(llm_map)} headlines "
            f"({n_hit} in training frame), usable dims: {usable}"
        )
    else:
        print("no validated LLM labels found; taxonomy targets = weak labels only")
    train_ds = HeadlineDataset(trn, tokenizer, max_length, llm_map, usable)
    val_ds = HeadlineDataset(val, tokenizer, max_length, llm_map, usable)
    batch_size = args.batch_size or params["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskHeadlineModel(model_name).to(device)
    weights = {
        "intensity": params["intensity_loss_weight"],
        "taxonomy": params["taxonomy_loss_weight"],
        "binary": params["binary_loss_weight"],
    }

    epochs = 1 if args.smoke else params["epochs"]
    total_steps = len(train_loader) * epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optim = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    warmup = max(1, int(total_steps * params["warmup_frac"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: (
            min(1.0, s / warmup) * max(0.0, 1 - max(0, s - warmup) / max(1, total_steps - warmup))
        ),
    )

    step = 0
    model.train()
    for _epoch in range(epochs):
        for batch in train_loader:
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = multitask_loss(
                out,
                {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k not in ("input_ids", "attention_mask")
                },
                weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            step += 1
            if step % 20 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} loss {loss.item():.4f}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    report = evaluate(model, val_loader, device)
    report["model_name"] = model_name
    report["steps"] = step
    report["n_train"] = len(trn)

    out_dir = Path(artifacts_dir("models")) / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "manipulens_config.json").write_text(
        json.dumps({"model_name": model_name, "dimensions": DIMENSIONS, "max_length": max_length})
    )

    report_path = REPO_ROOT / load_params()["artifacts"]["reports_dir"] / "transformer_eval.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"model -> {out_dir}\nreport -> {report_path}")
    return report


def load_trained(out_dir: Path | None = None) -> tuple[MultiTaskHeadlineModel, AutoTokenizer, dict]:
    out_dir = out_dir or Path(artifacts_dir("models")) / "transformer"
    cfg = json.loads((out_dir / "manipulens_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    model = MultiTaskHeadlineModel(cfg["model_name"])
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, tokenizer, cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    from manipulens.models import transformer as _canonical

    _canonical.main(sys.argv[1:])
