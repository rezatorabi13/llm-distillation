"""Typed experiment configuration loaded from YAML, with dotted-key overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TeacherConfig:
    name: str = "roneneldan/TinyStories-33M"


@dataclass
class StudentConfig:
    n_layer: int = 2
    n_embd: int = 128
    n_head: int = 4


@dataclass
class DataConfig:
    dataset: str = "roneneldan/TinyStories"
    num_train_stories: int = 30_000  # stories used for training (taken from the start of the train split)
    download_train_stories: int = 30_000  # stories downloaded and cached once
    download_val_stories: int = 2_000
    num_val_chunks: int = 1_000
    chunk_len: int = 64
    cache_dir: str = "~/.cache/lm-distill"


@dataclass
class TrainConfig:
    method: str = "distill"  # "distill" (teacher soft labels) or "baseline" (ground-truth next token)
    temperature: float = 1.0
    alpha: float = 1.0  # weight of the distillation term; (1 - alpha) goes to cross-entropy
    steps: int = 2_000
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batch_size: int = 32
    amp: bool = True  # bfloat16 autocast on CUDA
    seed: int = 0


@dataclass
class Config:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_name(self) -> str:
        t = self.train
        name = f"{t.method}_n{self.data.num_train_stories}"
        if t.method == "distill":
            name += f"_T{t.temperature:g}_a{t.alpha:g}"
        return f"{name}_s{t.seed}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {"teacher": TeacherConfig, "student": StudentConfig, "data": DataConfig, "train": TrainConfig}


def _coerce(value: Any, current: Any) -> Any:
    """Cast a CLI string override to the type of the field it replaces."""
    if not isinstance(value, str) or isinstance(current, str):
        return value
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes"}
    return type(current)(value)


def apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Return a copy of ``cfg`` with ``{"train.seed": 1, ...}`` style overrides applied."""
    cfg = dataclasses.replace(cfg, **{s: dataclasses.replace(getattr(cfg, s)) for s in _SECTIONS})
    for key, value in overrides.items():
        section, _, name = key.partition(".")
        if section not in _SECTIONS or not hasattr(getattr(cfg, section), name):
            raise KeyError(f"Unknown config key: {key!r}")
        sub = getattr(cfg, section)
        setattr(sub, name, _coerce(value, getattr(sub, name)))
    if cfg.train.method not in {"distill", "baseline"}:
        raise ValueError(f"train.method must be 'distill' or 'baseline', got {cfg.train.method!r}")
    return cfg


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Build a Config from defaults, an optional YAML file, and optional dotted-key overrides."""
    flat: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for section, values in raw.items():
            if section not in _SECTIONS:
                raise KeyError(f"Unknown config section {section!r} in {path}")
            flat.update({f"{section}.{k}": v for k, v in (values or {}).items()})
    flat.update(overrides or {})
    return apply_overrides(Config(), flat)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``["train.seed=1", "data.num_train_stories=300"]`` into a dict (values parsed as YAML)."""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {pair!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out
