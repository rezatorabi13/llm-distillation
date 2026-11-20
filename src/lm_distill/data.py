"""TinyStories download, tokenization, caching and fixed-length chunking."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import torch

from lm_distill.config import DataConfig

log = logging.getLogger(__name__)


def _tokenize_split(dataset: str, split: str, num_stories: int, tokenizer) -> torch.Tensor:
    from datasets import load_dataset

    stream = load_dataset(dataset, split=split, streaming=True)
    ids: list[int] = []
    for row in itertools.islice(stream, num_stories):
        ids += tokenizer(row["text"])["input_ids"] + [tokenizer.eos_token_id]  # EOS marks each story boundary
    return torch.tensor(ids, dtype=torch.long)


def load_tokens(cfg: DataConfig, tokenizer) -> dict[str, torch.Tensor]:
    """Return ``{"train": ids, "val": ids}``, downloading and caching on first use."""
    cache_dir = Path(cfg.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"tinystories_train{cfg.download_train_stories}_val{cfg.download_val_stories}.pt"
    if cache.exists():
        return torch.load(cache)
    log.info("Downloading and tokenizing %s (one time only) -> %s", cfg.dataset, cache)
    tokens = {
        "train": _tokenize_split(cfg.dataset, "train", cfg.download_train_stories, tokenizer),
        "val": _tokenize_split(cfg.dataset, "validation", cfg.download_val_stories, tokenizer),
    }
    torch.save(tokens, cache)
    return tokens


def first_n_stories(ids: torch.Tensor, n: int, eos_id: int) -> torch.Tensor:
    """Tokens of the first ``n`` stories (each story ends with ``eos_id``)."""
    ends = (ids == eos_id).nonzero().flatten()
    if n > len(ends):
        raise ValueError(f"Requested {n} stories but only {len(ends)} are available")
    return ids[: ends[n - 1].item() + 1]


def to_chunks(ids: torch.Tensor, chunk_len: int) -> torch.Tensor:
    """Split a 1-D token stream into ``(num_chunks, chunk_len)``, dropping the remainder."""
    usable = len(ids) // chunk_len * chunk_len
    return ids[:usable].view(-1, chunk_len)


def build_splits(cfg: DataConfig, tokenizer, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Train and validation chunks, already on ``device`` (the whole dataset fits in GPU memory)."""
    tokens = load_tokens(cfg, tokenizer)
    train_ids = first_n_stories(tokens["train"], cfg.num_train_stories, tokenizer.eos_token_id)
    train = to_chunks(train_ids, cfg.chunk_len)
    val = to_chunks(tokens["val"], cfg.chunk_len)[: cfg.num_val_chunks]
    return train.to(device), val.to(device)
