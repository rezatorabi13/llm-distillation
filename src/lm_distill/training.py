"""Training loop shared by the baseline and the distilled student."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from lm_distill.config import TrainConfig
from lm_distill.evaluation import evaluate
from lm_distill.losses import distillation_loss
from lm_distill.utils import autocast

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    history: list[dict[str, float]] = field(default_factory=list)
    train_seconds: float = 0.0

    @property
    def final(self) -> dict[str, float]:
        return self.history[-1]


def train_student(
    student: torch.nn.Module,
    train_chunks: torch.Tensor,
    val_chunks: torch.Tensor,
    cfg: TrainConfig,
    teacher: torch.nn.Module | None = None,
) -> TrainResult:
    """Train ``student`` in place and return its evaluation history.

    ``cfg.method == "baseline"``: cross-entropy on the real next token.
    ``cfg.method == "distill"``: ``distillation_loss`` against the teacher's logits.

    The teacher is used for evaluation in both cases. Batches are drawn from a
    generator seeded with ``cfg.seed``, so a baseline and a distilled run with the
    same seed see exactly the same data in the same order.
    """
    if teacher is None and cfg.method == "distill":
        raise ValueError("A teacher is required for method='distill'")
    device = train_chunks.device
    distill = cfg.method == "distill"

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    result = TrainResult()
    start = time.perf_counter()
    student.train()
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(len(train_chunks), (cfg.batch_size,), generator=generator, device=device)
        batch = train_chunks[idx]
        targets = batch[:, 1:]

        with autocast(device, cfg.amp):
            logits = student(batch).logits[:, :-1]
            if distill:
                with torch.no_grad():
                    teacher_logits = teacher(batch).logits[:, :-1]
        if distill:
            loss = distillation_loss(logits, teacher_logits, targets, cfg.temperature, cfg.alpha)
        else:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            metrics = evaluate(student, val_chunks, teacher, cfg.eval_batch_size, cfg.amp)
            result.history.append({"step": step, "train_loss": loss.item(), **metrics})
            log.info("step %5d | loss %.3f | val ppl %.2f", step, loss.item(), metrics["perplexity"])
    result.train_seconds = time.perf_counter() - start
    return result
