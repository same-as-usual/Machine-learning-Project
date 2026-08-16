"""Distill the multi-task transformer into a small student for edge deployment.

Teacher: models/artifacts/transformer/ (DeBERTa-v3-small multi-task)
Student: MiniLM-class encoder (default nreimers/MiniLM-L6-H384-uncased, 22M
params) with the same three heads, trained on the teacher's soft outputs over
the full headline pool (no gold labels needed — that's the point).

The student is what gets exported to ONNX INT8 (export_onnx.py) and eventually
shipped inside the browser extension.

Usage:
  python -m manipulens.models.distill            # full distillation
  python -m manipulens.models.distill --smoke    # tiny student, few steps

Writes: models/artifacts/student/, reports/distill_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from manipulens.config import REPO_ROOT, artifacts_dir, load_params
from manipulens.labeling.labeling_functions import DIMENSIONS
from manipulens.models.transformer import (
    HeadlineDataset,
    MultiTaskHeadlineModel,
    evaluate,
    load_trained,
)

SMOKE_STUDENT = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, CPU-friendly


@torch.no_grad()
def teacher_soft_targets(
    teacher, tokenizer, headlines: list[str], max_length: int, batch_size: int, device
) -> dict[str, torch.Tensor]:
    """Teacher logits over the pool (already-masked headlines)."""
    outs = {"intensity": [], "taxonomy": [], "binary": []}
    teacher.eval()
    for i in range(0, len(headlines), batch_size):
        enc = tokenizer(
            headlines[i : i + batch_size],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        o = teacher(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for k in outs:
            outs[k].append(o[k].cpu())
    return {k: torch.cat(v) for k, v in outs.items()}


def distill(args: argparse.Namespace) -> dict:
    from manipulens.models.transformer import cap_torch_threads

    cap_torch_threads()
    params = load_params()["distill"]
    tparams = load_params()["transformer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher, teacher_tok, cfg = load_trained()
    teacher.to(device)
    max_length = cfg["max_length"]

    # pool: all available headlines (labels unused — soft targets only)
    from manipulens.models.transformer import build_frame

    df = build_frame(smoke=args.smoke)
    from manipulens.models.neutralize import mask_entities

    pool = [mask_entities(h) for h in df["headline"].tolist()]

    student_name = args.student_name or (SMOKE_STUDENT if args.smoke else params["student_name"])
    student_tok = AutoTokenizer.from_pretrained(student_name)
    student = MultiTaskHeadlineModel(student_name).to(device)

    batch_size = params["batch_size"] if not args.smoke else 16
    print(f"computing teacher soft targets over {len(pool)} headlines ...")
    soft = teacher_soft_targets(teacher, teacher_tok, pool, max_length, batch_size, device)

    temp = params["temperature"]
    epochs = 1 if args.smoke else params["epochs"]
    optim = torch.optim.AdamW(student.parameters(), lr=params["lr"])
    n = len(pool)
    step = 0
    student.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            enc = student_tok(
                [pool[j] for j in idx.tolist()],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            out = student(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = (
                nn.functional.mse_loss(out["intensity"], soft["intensity"][idx].to(device))
                + nn.functional.kl_div(
                    nn.functional.logsigmoid(out["taxonomy"] / temp),
                    torch.sigmoid(soft["taxonomy"][idx].to(device) / temp),
                    reduction="batchmean",
                )
                + nn.functional.mse_loss(out["binary"], soft["binary"][idx].to(device))
            )
            loss.backward()
            optim.step()
            optim.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"distill step {step} loss {loss.item():.4f}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    # eval student on the same held-out frame as the teacher eval
    val = df.sample(frac=0.15, random_state=tparams["seed"]).reset_index(drop=True)
    val_ds = HeadlineDataset(val, student_tok, max_length)
    report = evaluate(student, DataLoader(val_ds, batch_size=batch_size), device)
    report["student_name"] = student_name
    report["teacher_name"] = cfg["model_name"]
    report["steps"] = step

    out_dir = Path(artifacts_dir("models")) / "student"
    out_dir.mkdir(parents=True, exist_ok=True)
    student_tok.save_pretrained(out_dir)
    torch.save(student.state_dict(), out_dir / "model.pt")
    (out_dir / "manipulens_config.json").write_text(
        json.dumps({"model_name": student_name, "dimensions": DIMENSIONS, "max_length": max_length})
    )
    report_path = REPO_ROOT / load_params()["artifacts"]["reports_dir"] / "distill_eval.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"student -> {out_dir}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    distill(args)


if __name__ == "__main__":
    from manipulens.models import distill as _canonical

    _canonical.main(sys.argv[1:])
