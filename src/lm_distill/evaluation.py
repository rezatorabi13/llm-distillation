"""Held-out metrics: perplexity on real text, KL to the teacher, and top-1 agreement."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from lm_distill.utils import autocast


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_chunks: torch.Tensor,
    teacher: torch.nn.Module | None = None,
    batch_size: int = 32,
    amp: bool = True,
) -> dict[str, float]:
    """Token-averaged metrics over ``val_chunks``.

    Returns ``perplexity`` (on the real next tokens) and, when a teacher is given,
    ``kl`` = mean KL(teacher || model) per token and ``agreement`` = fraction of
    positions where both models' most likely next token is the same.
    """
    was_training = model.training
    model.eval()
    ce = kl = agree = n = 0.0
    for i in range(0, len(val_chunks), batch_size):
        chunk = val_chunks[i : i + batch_size]
        targets = chunk[:, 1:]
        with autocast(chunk.device, amp):
            logits = model(chunk).logits[:, :-1].float()
            t_logits = teacher(chunk).logits[:, :-1].float() if teacher is not None else None
        ce += F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum").item()
        if t_logits is not None:
            kl += F.kl_div(
                F.log_softmax(logits, -1), F.log_softmax(t_logits, -1), log_target=True, reduction="sum"
            ).item()
            agree += (logits.argmax(-1) == t_logits.argmax(-1)).sum().item()
        n += targets.numel()
    model.train(was_training)

    metrics = {"perplexity": math.exp(ce / n)}
    if teacher is not None:
        metrics |= {"kl": kl / n, "agreement": agree / n}
    return metrics
