"""Teacher loading and student construction."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedModel

from lm_distill.config import StudentConfig


def load_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name)


def load_teacher(name: str, device: str | torch.device) -> PreTrainedModel:
    """Load a pretrained causal LM, frozen and in eval mode."""
    teacher = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    teacher.requires_grad_(False)
    return teacher


def build_student(cfg: StudentConfig, vocab_size: int, context_len: int, seed: int) -> GPT2LMHeadModel:
    """A small GPT-2 initialised from ``seed``, so paired runs start from identical weights."""
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=vocab_size, n_positions=context_len, n_layer=cfg.n_layer, n_embd=cfg.n_embd, n_head=cfg.n_head
    )
    return GPT2LMHeadModel(config)


def count_params(model: torch.nn.Module) -> int:
    """Unique parameters (tied input/output embeddings are counted once)."""
    return sum(p.numel() for p in model.parameters())
