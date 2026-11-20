"""Logit-based knowledge distillation for small causal language models."""

from lm_distill.config import Config, load_config
from lm_distill.losses import distillation_loss

__all__ = ["Config", "load_config", "distillation_loss"]
__version__ = "1.0.0"
