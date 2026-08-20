"""The three arms: they must differ in exactly one knob each, or the A/B measures two things."""

from __future__ import annotations

import pytest

from reranker.src.config import CUTS, SEL_PRM, SEL_RANDOM, TOKENS, load_config

TOKENS_ARM = "reranker/configs/prm_search_l6.yaml"
CUTS_ARM = "reranker/configs/prm_search_l6_cuts.yaml"
RANDOM_ARM = "reranker/configs/prm_search_l6_random.yaml"


def load(path):
    cfg = load_config(["--config", path])
    cfg.prm_search.validate()
    return cfg


@pytest.mark.parametrize("path", [TOKENS_ARM, CUTS_ARM, RANDOM_ARM])
def test_each_arm_loads_and_validates(path):
    load(path)


def test_the_arms_agree_on_the_beam_shape():
    shapes = [(load(p).prm_search.beam_width, load(p).prm_search.expand)
              for p in (TOKENS_ARM, CUTS_ARM, RANDOM_ARM)]
    assert len(set(shapes)) == 1


def test_the_cuts_arm_differs_only_in_the_cadence():
    base, cuts = load(TOKENS_ARM).prm_search, load(CUTS_ARM).prm_search
    assert base.advance == TOKENS and cuts.advance == CUTS
    assert base.selector == cuts.selector == SEL_PRM
    differing = {f for f in vars(base) if getattr(base, f) != getattr(cuts, f)}
    assert differing <= {"advance", "segment_max_tokens",
                         "code_steps_per_chunk", "prose_lines_per_chunk", "out_dir"}
    assert base.level == cuts.level and base.problems == cuts.problems


def test_the_random_arm_differs_only_in_the_selector():
    base, ctrl = load(TOKENS_ARM).prm_search, load(RANDOM_ARM).prm_search
    assert base.selector == SEL_PRM and ctrl.selector == SEL_RANDOM
    differing = {f for f in vars(base) if getattr(base, f) != getattr(ctrl, f)}
    assert differing <= {"selector", "out_dir"}


def test_every_arm_inherits_the_rollout_config_the_prm_was_trained_against():
    for path in (TOKENS_ARM, CUTS_ARM, RANDOM_ARM):
        cfg = load(path)
        assert cfg.prm_rollout.gen_model
        assert cfg.prm_rollout.max_length > 0
