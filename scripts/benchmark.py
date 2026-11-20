"""Compare teacher and student size and CPU latency.

Example:
    python scripts/benchmark.py checkpoints/distill_n30000_T1_a1_s0
"""

from __future__ import annotations

import argparse
import platform

import torch
from transformers import AutoModelForCausalLM

from lm_distill.benchmark import cpu_latency_ms, size_mb
from lm_distill.config import load_config
from lm_distill.models import count_params
from lm_distill.runner import setup
from lm_distill.utils import save_json, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("student", help="checkpoint directory of a trained student")
    parser.add_argument("--config", default="configs/distill.yaml")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="results/benchmark.json")
    args = parser.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    _, teacher, _, val_chunks = setup(cfg)
    student = AutoModelForCausalLM.from_pretrained(args.student)
    batch = val_chunks[: args.batch_size]

    report = {"batch": f"{args.batch_size} x {cfg.data.chunk_len} tokens", "cpu": platform.processor(),
              "torch_threads": torch.get_num_threads()}
    for name, model in [("teacher", teacher), ("student", student)]:
        report[name] = {"params": count_params(model), "size_mb": round(size_mb(model), 1),
                        "cpu_latency_ms": round(cpu_latency_ms(model, batch), 1)}
    t, s = report["teacher"], report["student"]
    report["ratios"] = {k: round(t[k] / s[k], 1) for k in ("params", "size_mb", "cpu_latency_ms")}
    save_json(report, args.out)
    print(report)


if __name__ == "__main__":
    main()
