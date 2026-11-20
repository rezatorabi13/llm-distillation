"""End-to-end check of training and evaluation on CPU with tiny random models (no downloads)."""

import pytest
import torch

from lm_distill.config import StudentConfig, TrainConfig
from lm_distill.evaluation import evaluate
from lm_distill.models import build_student
from lm_distill.training import train_student

VOCAB, CTX = 64, 16


@pytest.fixture(scope="module")
def setup():
    teacher = build_student(StudentConfig(n_layer=2, n_embd=32, n_head=2), VOCAB, CTX, seed=123).eval()
    teacher.requires_grad_(False)
    g = torch.Generator().manual_seed(0)
    return teacher, torch.randint(VOCAB, (256, CTX), generator=g), torch.randint(VOCAB, (32, CTX), generator=g)


def tiny_student(seed=0):
    return build_student(StudentConfig(n_layer=1, n_embd=16, n_head=2), VOCAB, CTX, seed)


def cfg(method, **kw):
    return TrainConfig(method=method, steps=60, batch_size=16, lr=3e-3, eval_every=30, amp=False, **kw)


def test_teacher_agrees_with_itself(setup):
    teacher, _, val = setup
    m = evaluate(teacher, val, teacher, amp=False)
    assert m["kl"] == pytest.approx(0.0, abs=1e-5)
    assert m["agreement"] == 1.0


def test_distillation_moves_student_toward_teacher(setup):
    teacher, train, val = setup
    student = tiny_student()
    before = evaluate(student, val, teacher, amp=False)["kl"]
    result = train_student(student, train, val, cfg("distill", temperature=2.0), teacher)
    assert [h["step"] for h in result.history] == [30, 60]
    assert result.final["kl"] < before


def test_same_seed_gives_identical_runs(setup):
    teacher, train, val = setup
    a = train_student(tiny_student(), train, val, cfg("baseline"), teacher).final
    b = train_student(tiny_student(), train, val, cfg("baseline"), teacher).final
    assert a == b


def test_distill_requires_teacher(setup):
    _, train, val = setup
    with pytest.raises(ValueError):
        train_student(tiny_student(), train, val, cfg("distill"), teacher=None)
