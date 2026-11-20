"""Train a single student.

Examples:
    python scripts/train.py --config configs/distill.yaml
    python scripts/train.py --config configs/baseline.yaml --set train.seed=1 data.num_train_stories=300
    python scripts/train.py --config configs/distill.yaml --save-dir checkpoints
"""

from __future__ import annotations

import argparse

from lm_distill.config import load_config, parse_overrides
from lm_distill.runner import run
from lm_distill.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config fields")
    parser.add_argument("--results-dir", default="results/runs")
    parser.add_argument("--save-dir", default=None, help="save the trained student here (Hugging Face format)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, parse_overrides(args.set))
    record = run(cfg, args.results_dir, args.save_dir)
    final = record["final"]
    print(f"\n{record['run_name']}: perplexity {final['perplexity']:.2f} | KL {final['kl']:.3f} | "
          f"agreement {final['agreement']:.1%} | {record['train_seconds']:.0f}s")


if __name__ == "__main__":
    main()
