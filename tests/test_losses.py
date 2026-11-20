import pytest
import torch
import torch.nn.functional as F

from lm_distill.losses import distillation_loss, soft_target_kl


@pytest.fixture
def logits():
    g = torch.Generator().manual_seed(0)
    student = torch.randn(2, 5, 50, generator=g)
    teacher = torch.randn(2, 5, 50, generator=g)
    return student, teacher, torch.randint(50, (2, 5), generator=g)


def test_zero_when_student_matches_teacher(logits):
    _, teacher, _ = logits
    for t in (1.0, 2.0, 4.0):
        assert distillation_loss(teacher.clone(), teacher, temperature=t).item() == pytest.approx(0.0, abs=1e-6)


def test_positive_when_distributions_differ(logits):
    student, teacher, _ = logits
    assert distillation_loss(student, teacher).item() > 0.1


def test_matches_reference_kl(logits):
    """Per-token KL(teacher || student), averaged over positions, times T^2."""
    student, teacher, _ = logits
    t = 3.0
    p = F.softmax(teacher / t, -1)
    expected = (p * (p.log() - F.log_softmax(student / t, -1))).sum(-1).mean() * t**2
    assert distillation_loss(student, teacher, temperature=t).item() == pytest.approx(expected.item(), rel=1e-5)


def test_alpha_endpoints(logits):
    student, teacher, targets = logits
    ce = F.cross_entropy(student.reshape(-1, 50), targets.reshape(-1))
    kl = soft_target_kl(student, teacher, 2.0)
    assert distillation_loss(student, teacher, targets, 2.0, alpha=0.0).item() == pytest.approx(ce.item(), rel=1e-6)
    assert distillation_loss(student, teacher, targets, 2.0, alpha=1.0).item() == pytest.approx(kl.item(), rel=1e-6)
    mixed = distillation_loss(student, teacher, targets, 2.0, alpha=0.3).item()
    assert mixed == pytest.approx(0.3 * kl.item() + 0.7 * ce.item(), rel=1e-5)


def test_t_squared_keeps_gradient_scale(logits):
    """Without the T^2 factor, gradients shrink ~1/T^2; with it they stay roughly constant."""
    student, teacher, _ = logits

    def grad_norm(t: float) -> float:
        s = student.clone().requires_grad_(True)
        distillation_loss(s, teacher, temperature=t).backward()
        return s.grad.norm().item()

    ratio = grad_norm(8.0) / grad_norm(16.0)
    assert ratio == pytest.approx(1.0, abs=0.1)


def test_validation(logits):
    student, teacher, _ = logits
    with pytest.raises(ValueError, match="Shape mismatch"):
        distillation_loss(student[:, :4], teacher)
    with pytest.raises(ValueError, match="targets are required"):
        distillation_loss(student, teacher, alpha=0.5)
    with pytest.raises(ValueError, match="alpha"):
        distillation_loss(student, teacher, alpha=1.5)
