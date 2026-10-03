"""The one shipped search config; every other run is it plus command-line overrides."""

from __future__ import annotations

from processkernel.config import SEL_PRM, SEL_RANDOM, TOKENS, load_config

BASE = "configs/prm_search.yaml"


def load(*overrides: str):
    cfg = load_config(["--config", BASE, *overrides])
    cfg.prm_search.validate()
    return cfg


def differing(a, b) -> set[str]:
    return {f for f in vars(a) if getattr(a, f) != getattr(b, f)}


def test_the_base_is_the_token_prm_search_with_a_4x4_beam():
    conf = load().prm_search
    assert (conf.advance, conf.selector) == (TOKENS, SEL_PRM)
    assert (conf.beam_width, conf.expand, conf.beam_groups) == (4, 4, 1)


def test_the_step_budget_covers_the_token_budget():
    cfg = load()
    assert cfg.prm_search.max_steps * cfg.prm_search.segment_max_tokens == \
        cfg.prm_rollout.max_new_tokens


def test_the_base_reads_the_staged_references_of_its_own_level():
    conf = load().prm_search
    assert conf.ref_dir.rstrip("/").endswith(f"level{conf.level}")


def test_the_base_scores_with_fused_attention():
    # ModelConfig defaults to eager, which materializes [batch, heads, seq, seq] and OOMs
    # next to a resident vLLM.
    assert load().model.attn_implementation == "sdpa"


def test_random_beam_is_one_override_away():
    prm, rnd = load(), load("prm_search.selector=random")
    assert rnd.prm_search.selector == SEL_RANDOM
    assert differing(prm.prm_search, rnd.prm_search) == {"selector"}


def test_a_single_problem_needs_a_trailing_comma_to_stay_a_string():
    # the override parser reads a bare "23" as an int
    assert load("prm_search.problems=23,").prm_search.problems == "23,"
