"""Run one or more experiment sweeps. Finished runs are skipped, so a sweep can be resumed.

Examples:
    python scripts/run_sweep.py configs/sweeps/main.yaml --save-dir checkpoints --save-seed 0
    python scripts/run_sweep.py configs/sweeps/scarce_data.yaml configs/sweeps/temperature.yaml
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import yaml

from lm_distill.config import load_config
from lm_distill.runner import free_memory, run, teacher_reference
from lm_distill.utils import save_json, setup_logging

log = logging.getLogger("sweep")


def expand(sweep_path: str):
    """Yield one Config per (base config, grid point, seed)."""
    spec = yaml.safe_load(Path(sweep_path).read_text())
    grid = spec.get("grid") or {}
    keys = list(grid)
    for base in spec["configs"]:
        for values in itertools.product(*(grid[k] for k in keys)):
            for seed in spec["seeds"]:
                overrides = dict(zip(keys, values, strict=True),**{"train.seed": seed})
                cfg = load_config(base, overrides)
                if cfg.train.method == "baseline" and "train.temperature" in overrides:
                    continue  # temperature has no effect on the baseline
                yield cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweeps", nargs="+", help="sweep YAML files")
    parser.add_argument("--results-dir", default="results/runs")
    parser.add_argument("--save-dir", default=None, help="save students to this directory")
    parser.add_argument("--save-seed", type=int, default=None, help="only save students with this seed")
    parser.add_argument("--force", action="store_true", help="re-run even if a result file exists")
    args = parser.parse_args()
    setup_logging()

    configs = {cfg.run_name: cfg for sweep in args.sweeps for cfg in expand(sweep)}  # de-duplicate shared runs
    todo = [c for name, c in configs.items() if args.force or not (Path(args.results_dir) / f"{name}.json").exists()]
    log.info("%d runs in sweep, %d to do", len(configs), len(todo))

    teacher_file = Path(args.results_dir).parent / "teacher.json"
    if todo and not teacher_file.exists():
        save_json(teacher_reference(todo[0]), teacher_file)

    for i, cfg in enumerate(todo, 1):
        log.info("[%d/%d] %s", i, len(todo), cfg.run_name)
        save = args.save_dir if args.save_seed in (None, cfg.train.seed) else None
        record = run(cfg, args.results_dir, save)
        log.info("[%d/%d] %s -> perplexity %.2f (%.0fs)", i, len(todo), cfg.run_name,
                 record["final"]["perplexity"], record["train_seconds"])
        free_memory()


if __name__ == "__main__":
    main()
