"""``prm.rollout.rollout``: one prefix -> K measured continuations (PLAN_v2 §6, job B).

Everything here runs against ``FakeBackend``. That is not a convenience: the two-pass seam
is the code most likely to be wrong and the least likely to be caught downstream, and the
fake reproduces the two vLLM behaviours that make it hard -- a truncated text whose token
ids run through the stop string, and a stop that fires mid-batch.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter

import pytest
import yaml

from kernel_gen.core.backend import FAKE_CHARS_PER_TOKEN, Backend, FakeBackend
from reranker.src.config import PRMRolloutConfig, RerankerConfig
from reranker.src.prm import build, corpus
from reranker.src.prm.rollout import prefixes, rollout, stage

RUN, TAG, SHARD = "a_run", "ar", "shard_00"
SYS, USER = "SYSTEM PROMPT", "USER PROMPT"
# The shape sampling.py produces: PLAN_PREFIX + plan + fence + code. Slicing it is what
# makes the reconstruction exact, so the fixture has to have that shape.
RAW = "## Plan\nfuse the two adds\n```python\nimport torch\n\n\nclass ModelNew:\n    pass\n"


def cfg(**over) -> PRMRolloutConfig:
    base = PRMRolloutConfig(
        run_tags={RUN: TAG},
        baseline_timing_json=__file__,
        K=2,
        min_rollouts=1,
        max_new_tokens=100,
        temperature=0.6,
        think_temperature=1.0,
    )
    for k, v in over.items():
        setattr(base, k, v)
    base.validate()
    return base


def prefix(**over) -> prefixes.Prefix:
    """One job-A row, defaulting to a code cut halfway through ``RAW``."""
    fields = dict(
        prefix_id=f"{TAG}__shard00__r0__p1__s0__k005",
        source="cut",
        run_name=RUN,
        run_tag=TAG,
        shard=SHARD,
        round=0,
        level=6,
        problem_id=1,
        sample_id=0,
        stem="level_6_problem_1_sample_0_kernel",
        cut_char=RAW.index("```python") + len("```python\nimport torch\n"),
        cut_index=5,
        cut_kind="code",
        n_cuts_total=20,
        rel_depth=0.25,
        list_key=f"{TAG}:6:1:0:5",
        split="train",
        selection="random",
        selection_score=None,
        K=2,
        min_rollouts=1,
    )
    fields.update(over)
    return prefixes.Prefix(**fields)


def source(raw=RAW) -> rollout.Source:
    return rollout.Source(system_prompt=SYS, prompt=USER, raw=raw)


def sources(*ps, raw=RAW) -> dict:
    return {rollout.source_key(p): source(raw) for p in (ps or (prefix(),))}


def part_row(*, sid=0, stem=None, sha="a" * 40, raw=RAW, **over):
    """A v1 part row -- only the fields job B reads, plus enough noise to be realistic."""
    row = {
        "run_name": RUN,
        "shard": SHARD,
        "round": 0,
        "level": 6,
        "problem_id": 1,
        "sample_id": sid,
        "stem": stem or f"level_6_problem_1_sample_{sid}_kernel",
        build.ROW_SHA1: sha,
        "prompt": USER,
        "raw": raw,
        "cuts": [10, 20],
        "cut_kinds": ["prose", "code"],
        "cut_index": [0, 1],
        "correct": True,
    }
    row.update(over)
    return row


def write_part(tmp_path, rows, name="a_run__shard_00__round0.jsonl", prompts=None):
    """A v1 out_dir: ``parts/<unit>.jsonl`` plus the ``system_prompts.json`` beside it."""
    parts = tmp_path / build.PARTS
    parts.mkdir(exist_ok=True)
    part = parts / name
    part.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / build.PROMPTS).write_text(json.dumps(prompts or {"a" * 40: SYS}))
    return str(part)


# --- prompt reconstruction: the campaign is invalid if this is off by one character -----


def test_reconstruct_is_the_rendered_chat_followed_by_the_raw_prefix():
    backend = FakeBackend()
    p = prefix()
    got = rollout.reconstruct(backend, p, source())
    assert got == backend.render_chat(SYS, USER) + RAW[: p.cut_char]


def test_reconstruct_at_the_end_of_raw_is_the_context_the_sampler_held():
    # The property the whole method rests on: raw IS PLAN_PREFIX + plan + seam + code, so
    # slicing it to its own length reproduces the string the two-pass sampler ended with.
    backend = FakeBackend()
    p = prefix(cut_char=len(RAW))
    assert rollout.reconstruct(backend, p, source()) == backend.render_chat(SYS, USER) + RAW


def test_reconstruct_appends_beam_text_after_the_cut():
    backend = FakeBackend()
    p = prefix(source="beam", beam_text="\n    return x\n")
    got = rollout.reconstruct(backend, p, source())
    assert got == backend.render_chat(SYS, USER) + RAW[: p.cut_char] + "\n    return x\n"


def test_a_cut_outside_raw_is_refused_rather_than_clamped():
    # A slice is not range-checked at either end: past the end Python returns the whole
    # completion, and a negative one counts back from it. Both silently measure a prefix
    # other than the one job A enumerated, and source_key compares (run, shard, round, stem),
    # so a part rebuilt with different `raw` keeps its key and reaches here.
    with pytest.raises(ValueError, match="cut_char"):
        rollout.prefix_text(prefix(cut_char=len(RAW) + 1), source())
    with pytest.raises(ValueError, match="cut_char"):
        rollout.prefix_text(prefix(cut_char=-20), source())
    with pytest.raises(ValueError, match="cut_char"):
        rollout.prefix_text(prefix(cut_char=-1), source())  # raw[:-1] drops one character
    rollout.prefix_text(prefix(cut_char=len(RAW)), source())  # the ends themselves are fine
    rollout.prefix_text(prefix(cut_char=0), source())


# --- resolving a prefix back to the texts v1 froze --------------------------------------


def test_load_sources_keys_on_the_run_shard_round_and_stem(tmp_path):
    part = write_part(tmp_path, [part_row(sid=0), part_row(sid=1)])
    got = rollout.load_sources(part, {"a" * 40: SYS})
    assert set(got) == {
        (RUN, SHARD, 0, "level_6_problem_1_sample_0_kernel"),
        (RUN, SHARD, 0, "level_6_problem_1_sample_1_kernel"),
    }
    assert got[rollout.source_key(prefix())] == rollout.Source(SYS, USER, RAW)


def test_a_prefix_and_its_v1_row_agree_on_the_source_key(tmp_path):
    part = write_part(tmp_path, [part_row()])
    assert rollout.source_key(prefix()) in rollout.load_sources(part, {"a" * 40: SYS})


def test_load_sources_names_a_system_prompt_sha_that_resolves_nowhere(tmp_path):
    part = write_part(tmp_path, [part_row(sha="b" * 40)])
    with pytest.raises(ValueError, match="b{40}"):
        rollout.load_sources(part, {"a" * 40: SYS})


def test_source_key_refuses_a_row_that_is_missing_a_field():
    # Strict, like prefixes._read_part: a tolerant .get turns a renamed v1 field into None,
    # every row in a part then collapses onto a key differing only by stem, and generate
    # resolves a prefix to another round's `raw` with the guard never firing.
    with pytest.raises(KeyError, match="round"):
        rollout.source_key({"run_name": "r", "shard": "s", "stem": "st"})


def test_load_prompts_reads_the_side_table_beside_the_parts_dir(tmp_path):
    write_part(tmp_path, [part_row()], prompts={"a" * 40: SYS, "c" * 40: "OTHER"})
    got = rollout.load_prompts(os.path.join(str(tmp_path), build.PARTS, "*.jsonl"))
    assert got == {"a" * 40: SYS, "c" * 40: "OTHER"}


def test_load_prompts_anchors_a_relative_glob_the_way_the_rest_of_the_config_does(monkeypatch):
    # prefixes.py resolves prm_rollout.parts_glob at both its call sites; this one has to
    # agree, or a relative glob reads against the process cwd under sbatch.
    seen = []
    monkeypatch.setattr(rollout, "_resolve", lambda p: seen.append(p) or "/nowhere")
    with pytest.raises(FileNotFoundError):
        rollout.load_prompts("data/prm_v2/parts/*.jsonl")
    assert seen == ["data/prm_v2/parts/*.jsonl"]


# --- the run a campaign claims to continue ----------------------------------------------


def write_source_run(tmp_path, run_name=RUN, shards=(SHARD,), **over):
    """A generation run dir as lintloop.sh leaves it: one config per shard, sampler and all."""
    body = {
        "model": "openai/gpt-oss-120b",
        "temperature": 0.6,
        "think_temperature": 1.0,
        "max_new_tokens": 16384,
    }
    body.update(over)
    run_dir = tmp_path / "runs" / run_name
    for shard in shards:
        (run_dir / shard).mkdir(parents=True)
        (run_dir / shard / rollout.GEN_CONFIG).write_text(yaml.safe_dump(body))
    return str(run_dir)


def write_v1_manifest(tmp_path, run_dirs):
    """v1's manifest, whose ``config.run_dirs`` is the only pointer back to the source run."""
    (tmp_path / build.MANIFEST).write_text(json.dumps({"config": {"run_dirs": list(run_dirs)}}))


def gen_cfg(tmp_path, **over):
    part = os.path.join(str(tmp_path), build.PARTS, "*.jsonl")
    return cfg(parts_glob=part, max_new_tokens=16384, **over)


def test_check_gen_model_returns_the_model_each_source_shard_was_written_by(tmp_path):
    write_v1_manifest(tmp_path, [write_source_run(tmp_path, shards=(SHARD, "shard_01"))])
    got = rollout.check_gen_model(gen_cfg(tmp_path), [(RUN, SHARD), (RUN, "shard_01")])
    assert got == {f"{RUN}/{SHARD}": "openai/gpt-oss-120b",
                   f"{RUN}/shard_01": "openai/gpt-oss-120b"}


def test_a_gen_model_that_did_not_write_the_prefixes_is_refused(tmp_path):
    # Three of the four candidate runs are DeepSeek and `gen_model` defaults to gpt-oss, so
    # switching run_tags and forgetting it sizes every budget with the wrong tokenizer and
    # measures V-hat under a model that never wrote the prefix -- invisibly, in both halves.
    write_v1_manifest(tmp_path, [write_source_run(tmp_path, model="deepseek-ai/DeepSeek-V4")])
    with pytest.raises(ValueError, match="deepseek-ai/DeepSeek-V4"):
        rollout.check_gen_model(gen_cfg(tmp_path), [(RUN, SHARD)])


def test_a_sampler_knob_that_differs_from_the_source_run_is_refused(tmp_path):
    # THINK_TEMP=0 runs are single-pass with no plan at all; continuing one of its prefixes
    # at think_temperature 1.0 samples a regime the completion was never generated in.
    write_v1_manifest(tmp_path, [write_source_run(tmp_path, think_temperature=0.0)])
    with pytest.raises(ValueError, match="think_temperature"):
        rollout.check_gen_model(gen_cfg(tmp_path), [(RUN, SHARD)])


def test_the_budget_cap_is_checked_against_the_source_run_too(tmp_path):
    # max_new_tokens is the ruler `budget` subtracts from: a cap larger than the source run's
    # hands a rollout more room than the completion it continues ever had.
    write_v1_manifest(tmp_path, [write_source_run(tmp_path, max_new_tokens=8192)])
    with pytest.raises(ValueError, match="max_new_tokens"):
        rollout.check_gen_model(gen_cfg(tmp_path), [(RUN, SHARD)])


def test_a_run_v1s_manifest_does_not_name_is_refused(tmp_path):
    write_v1_manifest(tmp_path, [write_source_run(tmp_path, run_name="other_run")])
    with pytest.raises(ValueError, match=RUN):
        rollout.check_gen_model(gen_cfg(tmp_path), [(RUN, SHARD)])


def test_a_shard_whose_generation_config_is_gone_is_refused(tmp_path):
    write_v1_manifest(tmp_path, [write_source_run(tmp_path, shards=(SHARD,))])
    with pytest.raises(FileNotFoundError, match="shard_09"):
        rollout.check_gen_model(gen_cfg(tmp_path), [(RUN, "shard_09")])


# --- the unit: what one part holds, and what one call is handed -------------------------


def test_a_units_name_is_the_v1_part_it_resolves_its_texts_from():
    # Job B looks a unit's texts up by this name, so it has to be build.part_name's -- both
    # source runs contain a shard_00, which is why the run name is in it at all.
    p = prefix()
    unit = corpus.Unit(RUN, "/runs/a_run", SHARD, 0, "attempts.jsonl", "eval.json")
    assert rollout.unit_name(p) + ".jsonl" == build.part_name(unit)


def test_units_group_prefixes_by_run_shard_and_round():
    # A part is one (run, shard, round) and so is a rollout part: mixing two rounds into one
    # would write a part that stage.py globs under a name naming only one of them.
    ps = [
        prefix(prefix_id="a", round=0),
        prefix(prefix_id="b", round=1),
        prefix(prefix_id="c", round=0),
        prefix(prefix_id="d", shard="shard_01", round=0),
    ]
    got = rollout.units(ps)
    assert list(got) == [
        f"{RUN}__{SHARD}__round0",
        f"{RUN}__{SHARD}__round1",
        f"{RUN}__shard_01__round0",
    ]
    assert [p.prefix_id for p in got[f"{RUN}__{SHARD}__round0"]] == ["a", "c"]


def test_a_unit_is_handed_to_generate_in_batches_that_lose_nothing():
    # Measured: one real unit is 14,005 prefixes -> 70,025 rollouts in a single call, ~770 MB
    # of pass-2 prompts held at once with nothing written until it all returns.
    ps = [prefix(prefix_id=str(i)) for i in range(7)]
    got = list(rollout.batches(ps, 3))
    assert [len(b) for b in got] == [3, 3, 1]
    assert [p.prefix_id for b in got for p in b] == [str(i) for i in range(7)]


def test_a_batch_size_of_zero_or_less_is_refused_rather_than_looping_forever():
    with pytest.raises(ValueError, match="prefixes_per_batch"):
        list(rollout.batches([prefix()], 0))


# --- the token budget -------------------------------------------------------------------


def test_budget_is_max_new_tokens_less_what_the_prefix_already_spent():
    assert rollout.budget(30, cfg(max_new_tokens=100)) == 70


def test_budget_is_never_negative():
    # raw is PLAN_PREFIX + plan + fence + code and each pass had its own max_new_tokens, so
    # a deep cut into a long two-pass completion can exceed the budget on its own.
    assert rollout.budget(140, cfg(max_new_tokens=100)) == 0
    assert rollout.budget(100, cfg(max_new_tokens=100)) == 0


def test_the_prefix_is_measured_over_the_generated_text_and_not_the_chat_header():
    # The budget bounds the *continuation* against the regime the source run generated
    # under. Counting the rendered prompt into it would shrink every rollout by the size of
    # a KernelBench problem statement, for no reason anyone could later see.
    p, src = prefix(), source()
    assert rollout.n_prefix_tokens(p, src, len) == len(RAW[: p.cut_char])


def test_a_beam_branch_counts_against_the_prefix_budget():
    p = prefix(source="beam", beam_text="\n    return x\n")
    assert rollout.n_prefix_tokens(p, source(), len) == p.cut_char + len(p.beam_text)


# --- the two continuation modes ---------------------------------------------------------


class Recorder(FakeBackend):
    """FakeBackend, remembering the sampling knobs each call was made with.

    The regime a rollout samples under is the whole point of the mode table and it leaves
    no other trace: temperature, stop and max_tokens are what the backend was *asked*, and
    the completion text that comes back looks the same either way.
    """

    def __init__(self, rules=None, default=""):
        super().__init__(rules, default)
        self.calls: list[dict] = []

    def complete_traced(self, prompts, **kw):
        self.calls.append(dict(kw, prompts=list(prompts)))
        return super().complete_traced(prompts, **kw)


class Plain(Backend):
    """Text and nothing else -- the base ``complete_traced`` is what wraps it."""

    def render_chat(self, system, user):
        return FakeBackend().render_chat(system, user)

    def complete(self, prompts, *, temperature, max_tokens, stop=None):
        return [KERNEL for _ in prompts]


def fake_tokens(text: str) -> int:
    return -(-len(text) // FAKE_CHARS_PER_TOKEN)


def run(backend, ps, conf=None, raw=RAW, count=len, counts=None):
    return rollout.generate(
        backend, list(ps), sources(*ps, raw=raw), conf or cfg(), count, counts
    )


def test_a_code_cut_is_one_pass_at_temperature_with_no_stop():
    backend = Recorder(default="\n    return x\n```\n")
    rows = run(backend, [prefix(cut_kind="code")])
    assert len(backend.calls) == 1
    assert backend.calls[0]["temperature"] == 0.6
    assert backend.calls[0]["stop"] is None
    assert [r.continuation for r in rows] == ["\n    return x\n```\n"] * 2


def test_a_prose_cut_is_two_passes_the_first_at_think_temperature_stopping_at_the_fence():
    # Not a preference: the source run generated the plan at think_temperature and the code
    # at temperature, so a prose cut continued at one temperature samples from a different
    # distribution than the prefix it is measuring.
    backend = Recorder(default="rest of the plan\n```python\nimport torch\n")
    run(backend, [prefix(cut_kind="prose", cut_char=12)])
    assert len(backend.calls) == 2
    assert backend.calls[0]["temperature"] == 1.0
    assert backend.calls[0]["stop"] == [rollout.CODE_FENCE]
    assert backend.calls[1]["temperature"] == 0.6
    assert backend.calls[1]["stop"] is None


def test_a_prose_pass_two_continues_from_the_bare_fence():
    # Mirroring sampling.py exactly: the fence the model *sees* carries no newline; the
    # newline below is inserted into the assembled text alone (KGEN-21).
    backend = Recorder(default="rest of the plan\n```python\nimport torch\n")
    p = prefix(cut_kind="prose", cut_char=12)
    run(backend, [p])
    head = backend.render_chat(SYS, USER) + RAW[:12]
    assert backend.calls[1]["prompts"][0] == head + "rest of the plan\n" + rollout.CODE_FENCE


def test_a_prose_continuation_is_the_plan_then_the_fence_then_the_code():
    backend = Recorder(
        rules=[("```python", "\nimport torch\n")], default="rest of the plan\n```python\n"
    )
    rows = run(backend, [prefix(cut_kind="prose", cut_char=12)])
    assert rows[0].continuation == "rest of the plan\n" + rollout.CODE_FENCE + "\nimport torch\n"


def test_the_seam_gains_a_newline_when_the_code_does_not_open_with_one():
    # The pair with the test above pins the conditional from both sides: glue the fence on
    # and ```pythonimport torch matches no fence at all, so the import sharing that line is
    # dropped (KGEN-21); always add the newline and the case above gains a blank line.
    backend = Recorder(
        rules=[("```python", "import torch\n")], default="rest of the plan\n```python\n"
    )
    rows = run(backend, [prefix(cut_kind="prose", cut_char=12)])
    assert rows[0].continuation.endswith(rollout.CODE_FENCE + "\nimport torch\n")
    assert "```pythonimport" not in rows[0].continuation


def test_an_unknown_cut_kind_is_refused_rather_than_sampled_as_code():
    with pytest.raises(ValueError, match="cut_kind"):
        run(Recorder(), [prefix(cut_kind="thinking")])


# --- a prose cut that is already past the pass-1/pass-2 seam -----------------------------
#
# `cut_kind` says what the text IS; the mode has to say which pass WROTE it, and the two part
# company after the seam. Pass 1 stops at CODE_FENCE and vLLM drops the stop string from the
# text, so a kept row's plan cannot contain one -- the first ```python in `raw` is the seam
# itself. Measured over three gpt-oss shards: 637 of 11,477 prose cuts (5.6%) are past it,
# 263 standing exactly at the fence and 374 in prose that pass 2 wrote after closing a block.

SEAM = RAW.index("```python") + len("```python\n")
# A ```text block is reachable inside a plan; a ```python one is not, because pass 1 would
# have stopped there. So this is what "prose after a closed fence, still pass 1" looks like.
PLAN_BLOCK = "## Plan\n```text\nshapes: 4x4\n```\nNow the kernel.\n```python\nimport torch\n"


def test_a_prose_cut_standing_at_the_seam_is_continued_as_code():
    backend = Recorder(default="\nimport torch\n")
    rows = run(backend, [prefix(cut_kind="prose", cut_char=SEAM)])
    assert len(backend.calls) == 1
    assert backend.calls[0]["temperature"] == 0.6
    assert backend.calls[0]["stop"] is None
    assert rows[0].continuation == "\nimport torch\n"
    assert rollout.CODE_FENCE not in rows[0].continuation
    assert rows[0].finish_reason["plan"] is None


def test_a_prose_cut_past_the_seam_is_continued_as_code_even_after_a_closed_block():
    # The 374 the open-fence rule missed: pass 2 wrote a kernel, closed the fence, and went
    # back to reasoning. That prose is still pass 2, and continuing it at the plan
    # temperature would sample it from a distribution it never came from.
    raw = RAW + "```\n\nBut tl.sum expects a vector; accumulate manually.\n"
    backend = Recorder(default="\nmore code\n")
    p = prefix(cut_kind="prose", cut_char=len(raw))
    run(backend, [p], raw=raw, conf=cfg(max_new_tokens=1000))
    assert len(backend.calls) == 1
    assert backend.calls[0]["temperature"] == 0.6


def test_a_prose_cut_after_a_block_the_plan_itself_opened_is_still_two_pass():
    # ```text is not the seam -- pass 1 only stops at ```python -- so this is still the plan.
    backend = Recorder(default="more plan\n```python\nimport torch\n")
    p = prefix(cut_kind="prose", cut_char=PLAN_BLOCK.index("Now the kernel.") + 10)
    run(backend, [p], raw=PLAN_BLOCK)
    assert len(backend.calls) == 2
    assert backend.calls[0]["temperature"] == 1.0


def test_a_prose_cut_before_the_seam_is_still_two_pass():
    backend = Recorder(default="more plan\n```python\nimport torch\n")
    run(backend, [prefix(cut_kind="prose", cut_char=12)])
    assert len(backend.calls) == 2


def test_a_beam_branch_can_carry_a_prose_cut_past_the_seam():
    # The prefix text is raw[:cut_char] + beam_text, so the branch counts as context here
    # exactly as it does for the budget.
    backend = Recorder(default="\nimport torch\n")
    p = prefix(cut_kind="prose", cut_char=12, source="beam", beam_text="\n```python\nimport os\n")
    run(backend, [p])
    assert len(backend.calls) == 1


def test_prose_cuts_past_the_seam_are_counted():
    # 5.6% of prose cuts on the real corpus -- a number the campaign report has to carry,
    # not a silent reclassification. Counted per prefix, not per rollout.
    counts: Counter = Counter()
    run(Recorder(default="x"), [prefix(cut_kind="prose", cut_char=SEAM)], counts=counts)
    assert counts["prose_cut_past_seam"] == 1


# --- batching: one call per pass, and never more tokens than a prompt's own budget -------


def test_the_K_rollouts_of_a_prefix_are_one_prompt_repeated():
    # vLLM's prefix cache collapses the shared prefill across them, which is the property
    # that makes K continuations cost roughly one prefill instead of K.
    # K off the prefix, not off job B's config: job A stamped what the campaign asked for,
    # and a config edited between the two jobs must not resize a fan-out already enumerated.
    backend = Recorder()
    run(backend, [prefix(K=4)])
    assert len(backend.calls[0]["prompts"]) == 4
    assert len(set(backend.calls[0]["prompts"])) == 1


def test_every_prefix_of_a_unit_goes_in_one_call_per_pass():
    backend = Recorder(default="x")
    ps = [
        prefix(prefix_id="p1", cut_kind="prose", cut_char=12),
        prefix(prefix_id="p2", cut_kind="code"),
        prefix(prefix_id="p3", cut_kind="prose", cut_char=12),
    ]
    run(backend, ps)
    assert len(backend.calls) == 2
    assert len(backend.calls[0]["prompts"]) == 4  # the two prose prefixes, K=2 each
    assert len(backend.calls[1]["prompts"]) == 6  # both prose pass 2 and the code cut


def test_no_prompt_is_given_more_tokens_than_its_own_prefix_left_it():
    # The bucket takes its members' MINIMUM, so the shallow prefix is handed the deep one's
    # budget rather than its own 92 -- that haircut is what BUDGET_QUANTUM's comment accepts.
    backend = Recorder(default="x")
    ps = [prefix(prefix_id="short", cut_char=8), prefix(prefix_id="long", cut_char=70)]
    run(backend, ps, cfg(max_new_tokens=100))
    header = len(backend.render_chat(SYS, USER))
    for call in backend.calls:
        for prompt in call["prompts"]:
            assert call["max_tokens"] <= 100 - (len(prompt) - header)
    assert [c["max_tokens"] for c in backend.calls] == [30]


def test_the_prompt_generate_sends_is_the_one_reconstruct_returns():
    # generate must not carry its own copy of the reconstruction: the tests that call this
    # file's premise load-bearing all go through `reconstruct`, and an inline second
    # implementation would leave them green while production drifted.
    backend = Recorder(default="x")
    p = prefix()
    run(backend, [p])
    assert backend.calls[0]["prompts"][0] == rollout.reconstruct(backend, p, source())


def test_the_budget_generate_uses_is_the_one_n_prefix_tokens_returns():
    backend = Recorder(default="x")
    p = prefix()
    rows = run(backend, [p])
    assert rows[0].n_prefix_tokens == rollout.n_prefix_tokens(p, source(), len)


def test_the_manifest_counters_count_prefixes_and_rollouts():
    # The two numbers the campaign manifest reports its fan-out with; without these, moving
    # either increment leaves every test green and the manifest wrong by a factor of K.
    counts: Counter = Counter()
    ps = [prefix(prefix_id="p1", K=2), prefix(prefix_id="p2", K=3)]
    rows = run(Recorder(default="x"), ps, counts=counts)
    assert (counts["prefixes"], counts["rollouts"]) == (2, 5)
    assert len(rows) == 5


def test_a_prefix_with_no_budget_left_is_dropped_before_it_is_sampled():
    backend = Recorder(default="x")
    counts: Counter = Counter()
    rows = run(backend, [prefix(cut_char=len(RAW))], cfg(max_new_tokens=4), counts=counts)
    assert rows == []
    assert counts["prefix_no_budget"] == 1
    assert backend.calls == []


# --- truncation: read off vLLM's own verdict, never inferred from the text ---------------


class Finishing(Recorder):
    """Recorder with a scripted ``finish_reason`` per pass; FakeBackend fixes it at "stop"."""

    def __init__(self, *reasons, **kw):
        super().__init__(**kw)
        self.reasons = list(reasons)

    def complete_traced(self, prompts, **kw):
        out = super().complete_traced(prompts, **kw)
        for completion in out:
            completion.finish_reason = self.reasons[len(self.calls) - 1]
        return out


def test_a_finished_code_rollout_is_ok():
    rows = run(Finishing("stop", default="x"), [prefix(cut_kind="code")])
    assert rows[0].truncation == "ok"
    assert rows[0].finish_reason == {"plan": None, "code": "stop"}


def test_a_code_rollout_that_hit_the_token_cap_is_truncated():
    # v1 §2 drops a forcibly-stopped completion because its label describes a kernel the
    # model never finished writing. The identical argument makes a truncated rollout
    # unusable, so job D removes it from the K rather than scoring it 0.
    rows = run(Finishing("length", default="x"), [prefix(cut_kind="code")])
    assert rows[0].truncation == "truncated"


def test_a_prose_rollout_whose_plan_hit_the_cap_is_truncated():
    backend = Finishing("length", "stop", default="plan\n```python\nimport torch\n")
    rows = run(backend, [prefix(cut_kind="prose", cut_char=12)])
    assert rows[0].truncation == "truncated"
    assert rows[0].finish_reason == {"plan": "length", "code": "stop"}


def test_a_prose_rollout_whose_code_hit_the_cap_is_truncated():
    backend = Finishing("stop", "length", default="plan\n```python\nimport torch\n")
    rows = run(backend, [prefix(cut_kind="prose", cut_char=12)])
    assert rows[0].truncation == "truncated"


def test_a_rollout_whose_backend_reported_no_finish_reason_is_unknown():
    rows = run(Finishing(None, default="x"), [prefix(cut_kind="code")])
    assert rows[0].truncation == "unknown"


def test_an_unterminated_fence_with_finish_reason_stop_is_kept():
    # THE regression guard. 99% of that run's gpt-oss completions end inside an open fence,
    # so inferring truncation from the text would throw the campaign away. The prefix here
    # opens a fence at RAW's ```python and the continuation never closes it.
    backend = Finishing("stop", default="\n    return x\n")
    rows = run(backend, [prefix(cut_kind="code")])
    whole = RAW[: prefix().cut_char] + rows[0].continuation
    assert "```python" in whole and whole.count("```") == 1
    assert rows[0].truncation == "ok"


# --- the rest of the row ----------------------------------------------------------------

KERNEL = "\nclass ModelNew(torch.nn.Module):\n    pass\n"


def test_code_is_extracted_from_the_prefix_and_the_continuation_together():
    # The fence opened before the cut and the imports are on the prefix's side of it, so
    # extracting from the continuation alone would submit a kernel missing its imports.
    rows = run(Recorder(default=KERNEL), [prefix(cut_kind="code")])
    assert "import torch" in rows[0].code
    assert "class ModelNew" in rows[0].code
    assert "## Plan" not in rows[0].code


def test_code_sha1_is_the_sha1_of_the_extracted_code():
    # The campaign-wide eval dedup key: two rollouts whose kernels are byte-identical are
    # evaluated once, and eval outweighs generation ~10:1.
    rows = run(Recorder(default=KERNEL), [prefix(cut_kind="code")])
    assert rows[0].code_sha1 == build._text_sha1(rows[0].code)
    assert rows[0].code_sha1 == rows[1].code_sha1


def test_n_prefix_tokens_is_what_the_budget_was_sized_with():
    p = prefix(cut_kind="code")
    rows = run(Recorder(default=KERNEL), [p])
    assert rows[0].n_prefix_tokens == len(RAW[: p.cut_char])


def test_n_gen_tokens_counts_the_tokens_of_every_pass():
    # Off the ids, not off the assembled text: pass 1's ids run through the stop string it
    # was truncated before, and the seam contains characters nobody generated.
    backend = Recorder(rules=[("```python", KERNEL)], default="rest of the plan\n```python\n")
    rows = run(backend, [prefix(cut_kind="prose", cut_char=12)])
    assert rows[0].n_gen_tokens == fake_tokens("rest of the plan\n```python") + fake_tokens(KERNEL)
    assert rows[0].n_gen_tokens != len(rows[0].continuation)


def test_n_gen_tokens_falls_back_to_the_tokenizer_when_the_backend_traced_nothing():
    # `complete_traced` has a working default in terms of `complete`, so a backend with no
    # internals to offer implements nothing -- and still has to produce a countable row.
    rows = run(Plain(), [prefix(cut_kind="code")])
    assert rows[0].n_gen_tokens == len(rows[0].continuation)


def test_staged_as_is_left_for_stage_py():
    rows = run(Recorder(default=KERNEL), [prefix(cut_kind="code")])
    assert rows[0].staged_as is None


def test_rollout_ids_are_unique_and_name_their_prefix_and_slot():
    ps = [prefix(prefix_id="p1"), prefix(prefix_id="p2")]
    rows = run(Recorder(default=KERNEL), ps)
    assert [r.rollout_id for r in rows] == ["p1__j00", "p1__j01", "p2__j00", "p2__j01"]
    assert [(r.prefix_id, r.j) for r in rows] == [("p1", 0), ("p1", 1), ("p2", 0), ("p2", 1)]


# --- the seams a batched job B can silently get wrong ------------------------------------

LONG = "## Plan\nfuse\n```python\nimport torch\n" + "y" * 2000


def test_a_completion_is_returned_to_the_prefix_it_was_generated_for():
    # Deep and shallow prefixes fall in different budget buckets, so they are two calls and
    # the deep one goes first. A batched job B that realigned by call order rather than by
    # slot would hand every prefix its neighbour's kernel, and nothing downstream could see it.
    backend = Recorder(rules=[("y" * 100, "DEEP")], default="SHALLOW")
    ps = [prefix(prefix_id="shallow", cut_char=30), prefix(prefix_id="deep", cut_char=1600)]
    rows = rollout.generate(
        backend, ps, sources(*ps, raw=LONG), cfg(max_new_tokens=3000), len
    )
    assert len(backend.calls) == 2
    assert backend.calls[0]["prompts"][0].endswith("y" * 100)   # the deep bucket first
    assert [r.prefix_id for r in rows] == ["shallow", "shallow", "deep", "deep"]
    assert [r.continuation for r in rows] == ["SHALLOW", "SHALLOW", "DEEP", "DEEP"]


def test_a_prefix_whose_v1_row_is_missing_names_it():
    # A bare KeyError here prints a 4-tuple and nothing about which campaign artifact is
    # stale; job B reads 144 parts and the prefixes were enumerated from a different build.
    p = prefix()
    with pytest.raises(KeyError, match=p.prefix_id):
        rollout.generate(Recorder(), [p], {}, cfg(), len)


def test_no_prefixes_is_no_calls():
    backend = Recorder()
    assert rollout.generate(backend, [], {}, cfg(), len) == []
    assert backend.calls == []


# --- job B: the driver that turns prefixes.jsonl into one part per unit ------------------


def a_unit(n=1, **over):
    """``n`` prefixes of one unit, with the v1 rows job B resolves their texts through."""
    ps = [prefix(prefix_id=f"p{i}", sample_id=i, **over) for i in range(n)]
    return ps, [part_row(sid=p.sample_id) for p in ps]


def campaign(tmp_path, ps, rows, **over):
    """A campaign on disk: v1's parts and manifest, the source run, and job A's output."""
    for p in ps:
        # Restated rather than read off unit_name: this is the convention job B has to find
        # its texts by, and a fixture built from the code under test would follow it anywhere.
        name = f"{p.run_name}__{p.shard}__round{p.round}.jsonl"
        mine = [r for r in rows if (r["run_name"], r["shard"], r["round"]) == (p.run_name, p.shard, p.round)]
        write_part(tmp_path, mine, name=name)
    write_v1_manifest(tmp_path, [write_source_run(tmp_path, shards=sorted({p.shard for p in ps}))])
    out = tmp_path / "campaign"
    out.mkdir(exist_ok=True)
    (out / prefixes.PREFIXES).write_text(
        "".join(json.dumps(dataclasses.asdict(p)) + "\n" for p in ps)
    )
    return RerankerConfig(prm_rollout=gen_cfg(tmp_path, out_dir=str(out), **over))


def parts_of(conf):
    """The parts themselves; each also has a .meta beside it, which stage.py globs past."""
    out_dir = conf.prm_rollout.out_dir
    return sorted(
        n for n in os.listdir(os.path.join(out_dir, stage.ROLLOUTS)) if n.endswith(".jsonl.gz")
    )


def two_units(tmp_path):
    """A campaign of exactly two units, sorted: ...round0 then ...round1."""
    ps, rows = a_unit(2)
    ps.append(prefix(prefix_id="q0", sample_id=0, round=1))
    rows.append(part_row(sid=0, round=1))
    return campaign(tmp_path, ps, rows)


def test_unit_names_are_the_arrays_index_space(tmp_path):
    conf = two_units(tmp_path)
    assert rollout.unit_names(conf.prm_rollout) == [
        "a_run__shard_00__round0",
        "a_run__shard_00__round1",
    ]


def many_units(tmp_path, n):
    """A campaign of ``n`` units, round0..round(n-1) -- enough to index past a small array."""
    ps, rows = a_unit(1)
    for r in range(1, n):
        ps.append(prefix(prefix_id=f"q{r}", sample_id=0, round=r))
        rows.append(part_row(sid=0, round=r))
    return campaign(tmp_path, ps, rows)


def test_main_selects_the_unit_at_the_array_tasks_own_index_not_a_stride(tmp_path, monkeypatch):
    # §6's absolute-index requirement, pinned at main() itself: a resubmit of --array=3,7
    # must hit units 3 and 7, never a re-sliced 0,1. run_rollouts is stubbed so nothing loads.
    conf = many_units(tmp_path, 8)
    names = rollout.unit_names(conf.prm_rollout)
    monkeypatch.setattr(rollout, "load_config", lambda argv: conf)

    seen = []
    manifest = {
        "rollouts": 0, "prefixes": 0, "units_generated": 0, "units_reused": 0,
        "prefix_caching": None, "counts": {},
    }
    monkeypatch.setattr(
        rollout, "run_rollouts", lambda cfg, only=None, **kw: seen.append(only) or manifest
    )

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    rollout.main(["--config", "unused"])
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    rollout.main(["--config", "unused"])
    assert seen == [names[3], names[7]]

    seen.clear()
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "8")  # one past the last of 8 units
    rollout.main(["--config", "unused"])
    assert seen == []


def test_only_generates_the_one_unit_it_names(tmp_path):
    # The array's whole contract: task i touches unit i and nothing else, so two tasks never
    # write one part and no unit is generated twice at temperature 0.6.
    conf = two_units(tmp_path)
    manifest = rollout.run_rollouts(
        conf, backend=FakeBackend(default="import torch\n"), count=len,
        only="a_run__shard_00__round1",
    )
    assert parts_of(conf) == ["a_run__shard_00__round1.jsonl.gz"]
    assert manifest["units_generated"] == 1


def test_a_unit_name_no_prefix_belongs_to_is_refused(tmp_path):
    # A typo in the array index space would otherwise be a task that silently generates
    # nothing and exits 0, leaving a gap staging only notices much later.
    conf = two_units(tmp_path)
    with pytest.raises(KeyError, match="round7"):
        rollout.run_rollouts(
            conf, backend=FakeBackend(default="import torch\n"), count=len,
            only="a_run__shard_00__round7",
        )


def test_the_driver_writes_one_part_per_unit_holding_K_rollouts_for_every_prefix(tmp_path):
    ps, rows = a_unit(2)
    ps.append(prefix(prefix_id="q0", sample_id=0, round=1))
    rows.append(part_row(sid=0, round=1))
    conf = campaign(tmp_path, ps, rows)
    manifest = rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)

    assert parts_of(conf) == ["a_run__shard_00__round0.jsonl.gz", "a_run__shard_00__round1.jsonl.gz"]
    got = stage.read_rollouts(stage.unit_path(conf.prm_rollout.out_dir, "a_run__shard_00__round0"))
    assert [r.rollout_id for r in got] == ["p0__j00", "p0__j01", "p1__j00", "p1__j01"]
    assert manifest["prefixes"] == 3 and manifest["rollouts"] == 6


def test_a_unit_whose_part_is_already_there_is_not_generated_again(tmp_path):
    # §12's rerun row: a second invocation re-runs no completed unit -- 0 new rollouts, and
    # not because the kernels dedup (they are sampled at temperature 0.6 and never repeat),
    # but because a part that exists means a unit that finished.
    ps, rows = a_unit(2)
    conf = campaign(tmp_path, ps, rows)
    rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)
    before = open(stage.unit_path(conf.prm_rollout.out_dir, "a_run__shard_00__round0"), "rb").read()

    second = Recorder()
    manifest = rollout.run_rollouts(conf, backend=second, count=len)
    assert second.calls == []
    assert manifest["units_generated"] == 0 and manifest["units_reused"] == 1
    assert open(stage.unit_path(conf.prm_rollout.out_dir, "a_run__shard_00__round0"), "rb").read() == before


def test_a_campaign_killed_between_units_resumes_instead_of_refusing(tmp_path, monkeypatch):
    # The wall-clock case, which is the normal one at ~14 GPU-h a unit: the manifest is the
    # only record of what the finished parts were sampled under, so writing it once at the
    # end leaves a resumed run unable to check itself and refusing outright.
    #
    # The kill is patched at _run_unit, not counted in the backend: generate() makes several
    # backend calls per unit (one per mode and token-budget bucket), so a call count would
    # not reliably land the fault *between* two units, which is the state under test.
    ps, rows = a_unit(2)
    ps.append(prefix(prefix_id="q0", sample_id=0, round=1))
    rows.append(part_row(sid=0, round=1))
    conf = campaign(tmp_path, ps, rows)

    real, done = rollout._run_unit, []

    def dies_after_one(*args, **kw):
        if done:
            raise RuntimeError("CUDA error: device-side assert triggered")
        done.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(rollout, "_run_unit", dies_after_one)

    with pytest.raises(RuntimeError):
        rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)
    monkeypatch.undo()
    out_dir = conf.prm_rollout.out_dir
    assert os.path.isfile(os.path.join(out_dir, rollout.ROLLOUT_MANIFEST)), \
        "a unit finished, so its settings must be on disk before the next job reads them"

    manifest = rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)
    assert manifest["units_generated"] == 1 and manifest["units_reused"] == 1


def test_a_campaign_killed_mid_loop_leaves_a_manifest_whose_counters_are_not_negative(tmp_path, monkeypatch):
    # Regression: _publish() used to pass len(todo), fixed for the whole invocation, while
    # _metas(out_dir) only grows as units actually finish. Three units, killed after the
    # first: the old code wrote units_generated=3, units_reused=1-3=-2.
    ps, rows = a_unit(1)
    ps.append(prefix(prefix_id="q0", sample_id=0, round=1))
    rows.append(part_row(sid=0, round=1))
    ps.append(prefix(prefix_id="r0", sample_id=0, round=2))
    rows.append(part_row(sid=0, round=2))
    conf = campaign(tmp_path, ps, rows)

    real, done = rollout._run_unit, []

    def dies_after_one(*args, **kw):
        if done:
            raise RuntimeError("CUDA error: device-side assert triggered")
        done.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(rollout, "_run_unit", dies_after_one)

    with pytest.raises(RuntimeError):
        rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)
    monkeypatch.undo()

    out_dir = conf.prm_rollout.out_dir
    manifest = json.load(open(os.path.join(out_dir, rollout.ROLLOUT_MANIFEST)))
    assert manifest["units_generated"] == len(parts_of(conf)) == 1
    assert manifest["units_reused"] == 0


def test_parts_with_no_manifest_at_all_say_so(tmp_path):
    # Only reachable for a campaign generated before the manifest moved into the loop, but
    # the old message read as a sampler drift from None and sent operators hunting a config edit.
    ps, rows = a_unit(2)
    conf = campaign(tmp_path, ps, rows)
    rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)
    os.remove(os.path.join(conf.prm_rollout.out_dir, rollout.ROLLOUT_MANIFEST))

    with pytest.raises(ValueError, match="no rollout_manifest.json"):
        rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)


def test_a_unit_is_handed_to_generate_in_batches_and_not_all_at_once(tmp_path, monkeypatch):
    # Measured on a real unit: 14,005 prefixes -> 70,025 rollouts in one call, ~770 MB of
    # prompts held with nothing written until every one of them returns.
    seen = []
    real = rollout.generate
    monkeypatch.setattr(
        rollout, "generate", lambda b, ps, *a, **k: seen.append(len(ps)) or real(b, ps, *a, **k)
    )
    ps, rows = a_unit(3)
    conf = campaign(tmp_path, ps, rows, prefixes_per_batch=2)
    rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)
    assert seen == [2, 1]
    part = stage.unit_path(conf.prm_rollout.out_dir, "a_run__shard_00__round0")
    assert len(stage.read_rollouts(part)) == 6


def test_a_gen_model_that_did_not_write_the_prefixes_stops_the_job_before_it_samples(tmp_path):
    # The check has to fire here and not in stage.py: by staging time the rollouts have been
    # sampled under the wrong tokenizer already, and the GPU hours are spent.
    ps, rows = a_unit(1)
    conf = campaign(tmp_path, ps, rows, gen_model="deepseek-ai/DeepSeek-V4-Flash")
    backend = Recorder()
    with pytest.raises(ValueError, match="DeepSeek-V4-Flash"):
        rollout.run_rollouts(conf, backend=backend, count=len)
    assert backend.calls == []
    assert not os.path.isdir(os.path.join(conf.prm_rollout.out_dir, stage.ROLLOUTS))


def test_a_unit_with_no_v1_part_to_read_its_texts_from_names_it(tmp_path):
    ps, rows = a_unit(1)
    conf = campaign(tmp_path, ps, rows)
    os.remove(os.path.join(str(tmp_path), build.PARTS, "a_run__shard_00__round0.jsonl"))
    with pytest.raises(FileNotFoundError, match="a_run__shard_00__round0"):
        rollout.run_rollouts(conf, backend=FakeBackend(), count=len)


def test_the_manifest_carries_the_counters_and_the_models_that_wrote_the_prefixes(tmp_path):
    # prose_cut_past_seam is "counted, not silent" only once something writes it down, and a
    # campaign's V-hat is only interpretable against the model that produced its prefixes.
    ps, rows = a_unit(1, cut_kind="prose", cut_char=len(RAW))   # past the seam: fence in the text
    conf = campaign(tmp_path, ps, rows)
    manifest = rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)
    assert manifest["counts"]["prose_cut_past_seam"] == 1
    assert manifest["gen_models"] == {"a_run/shard_00": "openai/gpt-oss-120b"}
    assert manifest["config"]["gen_model"] == "openai/gpt-oss-120b"
    assert manifest["units"]["a_run__shard_00__round0"]["rollouts"] == 2


def test_a_resumed_campaign_still_reports_the_counters_of_the_units_it_skipped(tmp_path):
    # The manifest is rewritten every invocation, so counters held only in memory would be
    # lost the moment a campaign is resumed -- and a resumed campaign is the normal case.
    ps, rows = a_unit(1, cut_kind="prose", cut_char=len(RAW))
    conf = campaign(tmp_path, ps, rows)
    first = rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)
    again = rollout.run_rollouts(conf, backend=Recorder(), count=len)
    assert again["counts"] == first["counts"]
    assert again["rollouts"] == first["rollouts"] and again["units"] == first["units"]


def test_a_reused_unit_is_refused_when_the_sampler_behind_it_has_moved(tmp_path):
    # A unit is reused on its part's name alone. Resume after a wall-clock kill with an edited
    # temperature and the campaign holds rollouts from two regimes, under one manifest that
    # claims the second for all of them. v1's build.py refuses the same way.
    ps, rows = a_unit(1)
    conf = campaign(tmp_path, ps, rows)
    rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)

    conf.prm_rollout.temperature = 0.9
    with pytest.raises(ValueError, match="temperature"):
        rollout.run_rollouts(conf, backend=Recorder(), count=len)


def test_a_reused_unit_survives_a_knob_that_changes_no_rollout(tmp_path):
    # The guard has to stay narrow, or an operator raising the batch size after an OOM is told
    # to throw the campaign away -- and starts deleting manifests instead.
    ps, rows = a_unit(2)
    conf = campaign(tmp_path, ps, rows)
    rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)

    conf.prm_rollout.prefixes_per_batch = 1
    conf.prm_rollout.num_workers = 4
    assert rollout.run_rollouts(conf, backend=Recorder(), count=len)["units_reused"] == 1


def test_the_driver_sizes_its_budgets_with_the_generation_models_tokenizer(tmp_path, monkeypatch):
    # base_model is the reranker's and bounds max_length; reaching for it here would size
    # every rollout's budget under a model that never writes one.
    asked = []
    monkeypatch.setattr(build, "token_counter", lambda name: asked.append(name) or len)
    ps, rows = a_unit(1)
    rollout.run_rollouts(campaign(tmp_path, ps, rows), backend=FakeBackend())
    assert asked == ["openai/gpt-oss-120b"]


def test_the_manifest_lands_beside_the_rollouts_it_describes(tmp_path):
    ps, rows = a_unit(1)
    conf = campaign(tmp_path, ps, rows)
    rollout.run_rollouts(conf, backend=FakeBackend(default="import torch\n"), count=len)
    path = os.path.join(conf.prm_rollout.out_dir, rollout.ROLLOUT_MANIFEST)
    assert json.load(open(path))["rollouts"] == 2


# --- prefix caching: the assumption the K-rollouts-per-prefix economics rest on ----------


class Cached(FakeBackend):
    """A backend shaped like vLLM's, down to where the cache setting is actually readable."""

    def __init__(self, on: bool):
        super().__init__(default="import torch\n")
        self.llm = type("LLM", (), {})()
        self.llm.llm_engine = type("Engine", (), {})()
        self.llm.llm_engine.vllm_config = type("Config", (), {})()
        self.llm.llm_engine.vllm_config.cache_config = type("Cache", (), {})()
        self.llm.llm_engine.vllm_config.cache_config.enable_prefix_caching = on


def test_prefix_caching_is_read_back_off_the_engine_rather_than_assumed():
    assert rollout.prefix_caching(Cached(True)) is True
    assert rollout.prefix_caching(Cached(False)) is False


def test_a_backend_that_cannot_be_asked_reports_nothing_rather_than_claiming_it_is_on():
    # FakeBackend has no engine; a driver test must not be told the cache is enabled.
    assert rollout.prefix_caching(FakeBackend()) is None


def test_the_manifest_records_whether_the_prefill_cache_was_actually_on(tmp_path):
    ps, rows = a_unit(1)
    conf = campaign(tmp_path, ps, rows)
    assert rollout.run_rollouts(conf, backend=FakeBackend(), count=len)["prefix_caching"] is None


def test_a_resumed_campaign_still_reports_the_cache_the_rollouts_were_sampled_under(tmp_path):
    # A resume generates nothing and so has no engine to ask, and the manifest is rewritten
    # every invocation -- so the record has to live with the unit, not with the run.
    ps, rows = a_unit(1)
    conf = campaign(tmp_path, ps, rows)
    assert rollout.run_rollouts(conf, backend=Cached(True), count=len)["prefix_caching"] is True
    again = rollout.run_rollouts(conf, backend=Recorder(), count=len)
    assert again["units_generated"] == 0 and again["prefix_caching"] is True


def test_a_run_with_prefix_caching_off_stops_rather_than_paying_K_times_for_one_prefill(tmp_path):
    # The K prompts of a prefix are byte-identical for exactly this reason; without the cache
    # the campaign silently costs K prefills per prefix instead of one.
    ps, rows = a_unit(1)
    conf = campaign(tmp_path, ps, rows)
    off = Cached(False)
    off.render_chat = FakeBackend().render_chat
    with pytest.raises(ValueError, match="prefix caching"):
        rollout.run_rollouts(conf, backend=off, count=len)


def test_the_budget_counter_is_built_from_the_generation_model(monkeypatch):
    # The one confusion the plan calls out by name: base_model is the reranker's tokenizer
    # and bounds max_length; gen_model is what actually writes the rollout.
    asked = []
    monkeypatch.setattr(build, "token_counter", lambda name: asked.append(name) or len)
    conf = cfg(gen_model="openai/gpt-oss-120b", base_model="Qwen/Qwen3-Reranker-4B")
    assert rollout.gen_counter(conf)("abcd") == 4
    assert asked == ["openai/gpt-oss-120b"]


# --- the single-pass regime: v6 runs generated with native thinking ----------------------


def single_pass_cfg(**over):
    """A campaign over a `think_temperature: 0` run -- one call, no plan pass."""
    return cfg(think_temperature=0.0, enable_thinking=True, **over)


def test_a_prose_cut_of_a_single_pass_run_is_still_one_call_at_temperature():
    # The whole point. A run sampled at think_temperature 0 made ONE call with no stop, so
    # it had no seam to be on either side of. Continuing a prose cut in two passes would
    # prefill a fence the model never wrote and resample the tail at a second temperature.
    backend = Recorder(default="\n    return x\n```\n")
    rows = run(backend, [prefix(cut_kind="prose", cut_char=12)], conf=single_pass_cfg())
    assert len(backend.calls) == 1
    assert backend.calls[0]["temperature"] == 0.6
    assert backend.calls[0]["stop"] is None
    assert all(r.finish_reason["plan"] is None for r in rows)


def test_a_single_pass_continuation_carries_no_plan_half():
    # `_continuation` and `_trace` both read plan=None as "one pass"; a prose cut that
    # reached the two-pass path would splice CODE_FENCE into the text it returns.
    backend = Recorder(default="\n    return x\n```\n")
    rows = run(backend, [prefix(cut_kind="prose", cut_char=12)], conf=single_pass_cfg())
    assert [r.continuation for r in rows] == ["\n    return x\n```\n"] * 2


def test_the_single_pass_shortcut_is_counted_not_silent():
    counts = Counter()
    run(backend := Recorder(default="x"), [prefix(cut_kind="prose", cut_char=12)],
        conf=single_pass_cfg(), counts=counts)
    assert counts["single_pass_cut"] == 1
    assert counts["prose_cut_past_seam"] == 0
    assert len(backend.calls) == 1


def test_a_two_pass_campaign_is_unaffected_by_the_shortcut():
    # The regression guard on the other side: think_temperature 1.0 keeps the plan pass.
    backend = Recorder(default="rest of the plan\n```python\nimport torch\n")
    run(backend, [prefix(cut_kind="prose", cut_char=12)], conf=cfg())
    assert len(backend.calls) == 2


# --- the sampler knobs the source run recorded -------------------------------------------


def test_a_run_generated_before_the_tail_cut_flags_existed_still_matches(tmp_path):
    # The kb6 corpus carries no enable_thinking/top_p/top_k key at all. A bare `.get` would
    # compare None against False/1.0/0 and refuse every campaign over it.
    write_v1_manifest(tmp_path, [write_source_run(tmp_path)])
    got = rollout.check_gen_model(gen_cfg(tmp_path), [(RUN, SHARD)])
    assert got == {f"{RUN}/{SHARD}": "openai/gpt-oss-120b"}


def test_a_tail_cut_that_differs_from_the_source_run_is_refused(tmp_path):
    # Qwen 0.95/20, MiniMax 0.95/40, Nemotron 0.95: three of the five v6 runs carry a cut
    # vLLM's defaults do not apply, so ignoring it samples a different distribution.
    write_v1_manifest(tmp_path, [write_source_run(tmp_path, top_p=0.95)])
    with pytest.raises(ValueError, match="top_p"):
        rollout.check_gen_model(gen_cfg(tmp_path), [(RUN, SHARD)])


def test_a_top_k_that_differs_from_the_source_run_is_refused(tmp_path):
    write_v1_manifest(tmp_path, [write_source_run(tmp_path, top_k=20)])
    with pytest.raises(ValueError, match="top_k"):
        rollout.check_gen_model(gen_cfg(tmp_path), [(RUN, SHARD)])


def test_native_thinking_that_differs_from_the_source_run_is_refused(tmp_path):
    # render_chat branches on it: at False it CLOSES the block a native-thinking template
    # opens, so the prefix would be reconstructed under a head its generation never had.
    write_v1_manifest(tmp_path, [
        write_source_run(tmp_path, enable_thinking=True, think_temperature=0.0)
    ])
    with pytest.raises(ValueError, match="enable_thinking"):
        rollout.check_gen_model(gen_cfg(tmp_path, think_temperature=0.0), [(RUN, SHARD)])


def test_the_backend_is_built_with_the_source_runs_whole_sampler(monkeypatch):
    # Left to VLLMBackend's defaults these are False/1.0/0 -- the off-positions -- and
    # nothing downstream can see that the rollouts were sampled in another regime.
    import kernel_gen.core.backend as backend_mod

    seen = {}

    class Spy:
        def __init__(self, model, **kw):
            seen.update(kw, model=model)

    monkeypatch.setattr(backend_mod, "VLLMBackend", Spy)
    rollout._backend(single_pass_cfg(top_p=0.95, top_k=20))
    assert seen["enable_thinking"] is True
    assert seen["top_p"] == 0.95
    assert seen["top_k"] == 20
