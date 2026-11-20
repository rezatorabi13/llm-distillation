"""Distillation objective (Hinton et al., 2015) applied per token position."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_target_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Mean over token positions of KL(teacher_T || student_T), scaled by T^2.

    The T^2 factor keeps gradient magnitudes roughly independent of the temperature,
    so the same learning rate works across temperatures.
    """
    vocab = student_logits.size(-1)
    s = F.log_softmax(student_logits.reshape(-1, vocab).float() / temperature, dim=-1)
    t = F.log_softmax(teacher_logits.reshape(-1, vocab).float() / temperature, dim=-1)
    return F.kl_div(s, t, log_target=True, reduction="batchmean") * temperature**2


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor | None = None,
    temperature: float = 1.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """``alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(student, targets)``.

    Args:
        student_logits: ``(..., vocab)`` logits from the student.
        teacher_logits: ``(..., vocab)`` logits from the frozen teacher, same shape.
        targets: ``(...)`` ground-truth token ids; required only when ``alpha < 1``.
        temperature: softening temperature ``T`` applied to both distributions.
        alpha: weight of the distillation term. ``alpha=1`` is pure distillation:
            the student never sees the ground-truth next token.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Shape mismatch: student {tuple(student_logits.shape)} vs teacher {tuple(teacher_logits.shape)}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    loss = student_logits.new_zeros((), dtype=torch.float32)
    if alpha > 0:
        loss = loss + alpha * soft_target_kl(student_logits, teacher_logits, temperature)
    if alpha < 1:
        if targets is None:
            raise ValueError("targets are required when alpha < 1")
        ce = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)).float(), targets.reshape(-1))
        loss = loss + (1 - alpha) * ce
    return loss
