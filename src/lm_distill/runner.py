"""End-to-end run: build data and models from a Config, train, evaluate, save."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import torch

from lm_distill.config import Config
from lm_distill.data import build_splits
from lm_distill.evaluation import evaluate
from lm_distill.models import build_student, count_params, load_teacher, load_tokenizer
from lm_distill.training import train_student
from lm_distill.utils import get_device, save_json, set_seed

log = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _teacher(name: str, device: str):
    return load_teacher(name, device)


@lru_cache(maxsize=2)
def _tokenizer(name: str):
    return load_tokenizer(name)


@lru_cache(maxsize=8)
def _splits(cfg_key: tuple, name: str, device: str):
    from lm_distill.config import DataConfig

    return build_splits(DataConfig(*cfg_key), _tokenizer(name), device)


def setup(cfg: Config):
    """Tokenizer, frozen teacher and data splits (cached across runs in one process)."""
    device = str(get_device())
    tokenizer = _tokenizer(cfg.teacher.name)
    teacher = _teacher(cfg.teacher.name, device)
    data_key = tuple(getattr(cfg.data, f) for f in cfg.data.__dataclass_fields__)
    train_chunks, val_chunks = _splits(data_key, cfg.teacher.name, device)
    return tokenizer, teacher, train_chunks, val_chunks


def run(cfg: Config, results_dir: str | Path = "results/runs", save_dir: str | Path | None = None) -> dict:
    """Train one student as described by ``cfg`` and write ``<results_dir>/<run_name>.json``."""
    set_seed(cfg.train.seed)
    tokenizer, teacher, train_chunks, val_chunks = setup(cfg)
    vocab_size = teacher.config.vocab_size  # student logits must align with the teacher's
    student = build_student(cfg.student, vocab_size, cfg.data.chunk_len, cfg.train.seed).to(train_chunks.device)
    log.info(
        "%s | %d train chunks | student %.2fM params", cfg.run_name, len(train_chunks), count_params(student) / 1e6
    )

    result = train_student(student, train_chunks, val_chunks, cfg.train, teacher)
    record = {
        "run_name": cfg.run_name,
        "config": cfg.to_dict(),
        "num_train_chunks": len(train_chunks),
        "student_params": count_params(student),
        "train_seconds": round(result.train_seconds, 1),
        "final": result.final,
        "history": result.history,
    }
    save_json(record, Path(results_dir) / f"{cfg.run_name}.json")
    if save_dir is not None:
        out = Path(save_dir) / cfg.run_name
        student.save_pretrained(out)
        tokenizer.save_pretrained(out)
        log.info("Saved student to %s", out)
    return record


def teacher_reference(cfg: Config) -> dict:
    """Teacher metrics on the same validation chunks (its KL to itself is 0 by definition)."""
    tokenizer, teacher, _, val_chunks = setup(cfg)
    metrics = evaluate(teacher, val_chunks, None, cfg.train.eval_batch_size, cfg.train.amp)
    return {"name": cfg.teacher.name, "params": count_params(teacher), **metrics}


def free_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
