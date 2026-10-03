"""``PRMSearchConfig``: the knobs the beam search is tuned by, and what they refuse."""

from __future__ import annotations

import pytest

from processkernel.config import CUTS, SEL_RANDOM, TOKENS, PRMSearchConfig, RerankerConfig


def test_the_defaults_are_the_four_by_four_beam_the_plan_specifies():
    conf = PRMSearchConfig()
    assert (conf.beam_width, conf.expand) == (4, 4)
    assert conf.advance == TOKENS
    assert conf.segment_max_tokens == 256
    assert conf.min_distinct_parents == 2


def test_it_is_reachable_from_the_top_level_config():
    assert isinstance(RerankerConfig().prm_search, PRMSearchConfig)


@pytest.mark.parametrize(
    "over",
    [
        {"beam_width": 0},
        {"expand": 0},
        {"max_steps": 0},
        {"segment_max_tokens": 0},
        {"prose_lines_per_chunk": 0},
        {"code_steps_per_chunk": 0},
        {"advance": "lines"},
        {"selector": "orm"},
    ],
)
def test_validate_refuses_a_nonsense_setting(over):
    conf = PRMSearchConfig(prm_checkpoint="/ckpt", **over)
    with pytest.raises(ValueError):
        conf.validate()


def test_the_beam_width_must_split_evenly_into_sub_beams():
    conf = PRMSearchConfig(beam_width=4, beam_groups=3, prm_checkpoint="/ckpt")
    with pytest.raises(ValueError, match="multiple of"):
        conf.validate()


def test_a_diversity_floor_above_the_beam_width_is_unsatisfiable():
    conf = PRMSearchConfig(beam_width=2, min_distinct_parents=3, prm_checkpoint="/ckpt")
    with pytest.raises(ValueError, match="unsatisfiable"):
        conf.validate()


def test_the_prm_selector_needs_a_checkpoint_but_the_random_control_does_not():
    with pytest.raises(ValueError, match="prm_checkpoint"):
        PRMSearchConfig().validate()
    PRMSearchConfig(selector=SEL_RANDOM).validate()


def test_the_cuts_policy_validates_the_same_way():
    PRMSearchConfig(advance=CUTS, code_steps_per_chunk=16, prose_lines_per_chunk=16,
                    prm_checkpoint="/ckpt").validate()


def test_vllm_leaves_room_for_the_two_scorers_on_the_same_card():
    """0.92 is vLLM's default and would leave ~6.4 GB of 80 for the PRM, the ORM and their
    activations. The search's default has to be lower or the two scorers race the KV cache."""
    assert PRMSearchConfig().gpu_memory_utilization < 0.92


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_validate_refuses_a_utilization_outside_the_open_unit_interval(bad):
    conf = PRMSearchConfig(prm_checkpoint="/ckpt", gpu_memory_utilization=bad)
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        conf.validate()
