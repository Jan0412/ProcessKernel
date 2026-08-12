"""``prm.rollout.rollout``: one prefix -> K measured continuations (PLAN_v2 §6, job B).

Everything here runs against ``FakeBackend``. That is not a convenience: the two-pass seam
is the code most likely to be wrong and the least likely to be caught downstream, and the
fake reproduces the two vLLM behaviours that make it hard -- a truncated text whose token
ids run through the stop string, and a stop that fires mid-batch.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import pytest

from kernel_gen.core.backend import FAKE_CHARS_PER_TOKEN, Backend, FakeBackend
from reranker.src.config import PRMRolloutConfig
from reranker.src.prm import build
from reranker.src.prm.rollout import prefixes, rollout

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


def test_the_budget_counter_is_built_from_the_generation_model(monkeypatch):
    # The one confusion the plan calls out by name: base_model is the reranker's tokenizer
    # and bounds max_length; gen_model is what actually writes the rollout.
    asked = []
    monkeypatch.setattr(build, "token_counter", lambda name: asked.append(name) or len)
    conf = cfg(gen_model="openai/gpt-oss-120b", base_model="Qwen/Qwen3-Reranker-4B")
    assert rollout.gen_counter(conf)("abcd") == 4
    assert asked == ["openai/gpt-oss-120b"]
