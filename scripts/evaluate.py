"""Evaluate a saved student (or the teacher) on the validation chunks.

Examples:
    python scripts/evaluate.py checkpoints/distill_n30000_T1_a1_s0
    python scripts/evaluate.py teacher
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoModelForCausalLM

from lm_distill.config import load_config
from lm_distill.evaluation import evaluate
from lm_distill.models import count_params
from lm_distill.runner import setup
from lm_distill.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="checkpoint directory, or 'teacher'")
    parser.add_argument("--config", default="configs/distill.yaml", help="supplies teacher and data settings")
    args = parser.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    _, teacher, _, val_chunks = setup(cfg)
    if args.model == "teacher":
        model, ref = teacher, None
    else:
        model, ref = AutoModelForCausalLM.from_pretrained(args.model).to(val_chunks.device), teacher
    metrics = evaluate(model, val_chunks, ref, cfg.train.eval_batch_size, cfg.train.amp)
    print(json.dumps({"model": args.model, "params": count_params(model), **metrics}, indent=2))


if __name__ == "__main__":
    main()
