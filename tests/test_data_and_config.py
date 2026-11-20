import pytest
import torch

from lm_distill.config import load_config, parse_overrides
from lm_distill.data import first_n_stories, to_chunks

EOS = 99


def test_to_chunks_drops_remainder():
    chunks = to_chunks(torch.arange(10), 4)
    assert chunks.shape == (2, 4)
    assert chunks[1].tolist() == [4, 5, 6, 7]


def test_first_n_stories_cuts_at_story_boundary():
    ids = torch.tensor([1, 2, EOS, 3, EOS, 4, 5, 6, EOS])
    assert first_n_stories(ids, 1, EOS).tolist() == [1, 2, EOS]
    assert first_n_stories(ids, 2, EOS).tolist() == [1, 2, EOS, 3, EOS]
    with pytest.raises(ValueError):
        first_n_stories(ids, 4, EOS)


def test_shipped_configs_load():
    distill = load_config("configs/distill.yaml")
    baseline = load_config("configs/baseline.yaml")
    assert distill.train.method == "distill" and baseline.train.method == "baseline"
    # The two configs must differ only in the training objective
    assert distill.student == baseline.student and distill.data == baseline.data
    assert distill.run_name == "distill_n30000_T1_a1_s0"
    assert baseline.run_name == "baseline_n30000_s0"


def test_overrides_are_typed():
    overrides = parse_overrides(["train.seed=3", "train.temperature=2", "train.amp=false"])
    cfg = load_config("configs/distill.yaml", overrides)
    assert cfg.train.seed == 3 and cfg.train.temperature == 2.0 and cfg.train.amp is False
    with pytest.raises(KeyError):
        load_config(None, {"train.nope": 1})
    with pytest.raises(ValueError):
        load_config(None, {"train.method": "other"})
