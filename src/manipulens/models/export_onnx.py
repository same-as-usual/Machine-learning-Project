"""Export a trained ManipuLens transformer (teacher or student) to ONNX,
apply dynamic INT8 quantization, and benchmark parity/latency/size.

This is the artifact chain for edge deployment: the quantized model serves the
API without torch and runs in-browser via ONNX Runtime Web.

Usage:
  python -m manipulens.models.export_onnx --model-dir models/artifacts/student
  python -m manipulens.models.export_onnx --model-dir models/artifacts/transformer

Writes: <model-dir>/model.onnx, <model-dir>/model.int8.onnx,
        reports/onnx_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from manipulens.config import REPO_ROOT, load_params
from manipulens.models.neutralize import mask_entities
from manipulens.models.transformer import load_trained

BENCH_HEADLINES = [
    "You Won't Believe What This Senator Said Next",
    "Senate passes infrastructure bill in 69-30 vote",
    "27 INSANE budget facts!! #9 will blow your mind",
    "Federal Reserve holds interest rates steady",
    "The silent killer hiding in your kitchen",
    "Study suggests link between diet and sleep quality",
]


class OnnxWrapper(torch.nn.Module):
    """Flatten the dict output to a tuple for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask)
        return (
            torch.sigmoid(out["intensity"]),
            torch.sigmoid(out["taxonomy"]),
            torch.sigmoid(out["binary"]),
        )


def export(model_dir: Path) -> dict:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from manipulens.models.transformer import cap_torch_threads

    cap_torch_threads()

    model, tokenizer, cfg = load_trained(model_dir)
    wrapper = OnnxWrapper(model).eval()
    max_length = cfg["max_length"]

    enc = tokenizer(
        BENCH_HEADLINES[0],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    onnx_path = model_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (enc["input_ids"], enc["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["intensity", "taxonomy", "binary"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "intensity": {0: "batch"},
            "taxonomy": {0: "batch"},
            "binary": {0: "batch"},
        },
        opset_version=17,
        # legacy tracer: the dynamo exporter's graph currently breaks
        # onnxruntime quantize_dynamic shape inference
        dynamo=False,
    )

    int8_path = model_dir / "model.int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)

    report: dict = {
        "model_dir": str(model_dir),
        "model_name": cfg["model_name"],
        "fp32_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "int8_mb": round(int8_path.stat().st_size / 1e6, 2),
    }

    # parity + latency
    def encode(headlines: list[str]):
        masked = [mask_entities(h) for h in headlines]
        e = tokenizer(
            masked,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="np",
        )
        return {
            "input_ids": e["input_ids"].astype(np.int64),
            "attention_mask": e["attention_mask"].astype(np.int64),
        }

    feeds = encode(BENCH_HEADLINES)
    with torch.no_grad():
        t_out = wrapper(torch.tensor(feeds["input_ids"]), torch.tensor(feeds["attention_mask"]))
    torch_intensity = t_out[0].numpy()

    for tag, path in (("fp32", onnx_path), ("int8", int8_path)):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        o_intensity = sess.run(["intensity"], feeds)[0]
        report[f"{tag}_max_abs_diff_vs_torch"] = float(np.abs(o_intensity - torch_intensity).max())

        single = encode([BENCH_HEADLINES[0]])
        sess.run(None, single)  # warmup
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            sess.run(None, single)
            times.append((time.perf_counter() - t0) * 1000)
        report[f"{tag}_p50_ms"] = round(float(np.percentile(times, 50)), 2)
        report[f"{tag}_p95_ms"] = round(float(np.percentile(times, 95)), 2)

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args(argv)

    report = export(Path(args.model_dir).resolve())
    report_path = REPO_ROOT / load_params()["artifacts"]["reports_dir"] / "onnx_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
