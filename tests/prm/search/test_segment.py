"""``segment.advance``: one step of generation over every live candidate, batched."""

from __future__ import annotations

from processkernel.generation.core.backend import FakeBackend
from processkernel.generation.core.model import Problem
from processkernel.generation.core.sampling import CODE_FENCE
from processkernel.config import CUTS, PRMRolloutConfig, PRMSearchConfig
from processkernel.prm.data.chunks import CODE, PROSE, live_cuts
from processkernel.prm.search import segment
from processkernel.prm.search.candidate import Candidate

PROBLEM = Problem(level=6, problem_id=12, name="12_Thing.py", ref_arch_src="import torch\n")


def count(text: str) -> int:
    """A stand-in gen tokenizer: four characters a token, as FakeBackend also assumes."""
    return (len(text) + 3) // 4


def cand(text="## Plan\n", mode=PROSE, **over) -> Candidate:
    base = dict(problem=PROBLEM, head="<HEAD>", prompt="<PROMPT>", text=text, mode=mode)
    return Candidate(**{**base, **over})


def confs(**over):
    search = PRMSearchConfig(prm_checkpoint="/ckpt", **over)
    rollout = PRMRolloutConfig(max_new_tokens=4096, temperature=0.8, think_temperature=1.0)
    return search, rollout


class Finishing(FakeBackend):
    """FakeBackend with a scripted finish_reason -- plain FakeBackend fixes it at "stop",
    so a code completion always looks naturally finished (see test_rollout.py:592)."""

    def __init__(self, reason, **kw):
        super().__init__(**kw)
        self.reason = reason

    def complete_traced(self, prompts, **kw):
        out = super().complete_traced(prompts, **kw)
        for completion in out:
            completion.finish_reason = self.reason
        return out


class Recorder(FakeBackend):
    """FakeBackend, remembering each call's kwargs -- test_rollout.py:350 has the original,
    logging more than this needs; here only the kwargs (esp. max_tokens) matter."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls: list[dict] = []

    def complete_traced(self, prompts, **kw):
        self.calls.append({"prompts": list(prompts), **kw})
        return super().complete_traced(prompts, **kw)


def test_a_plan_that_reaches_the_fence_flips_to_code_and_keeps_the_bare_fence():
    # Distinguishable per pass, as in the seam test below: a single "<HEAD>"-keyed rule
    # would fire for both calls and the code pass's echo would already contain the
    # expected substring, passing even if the bare-fence append were deleted entirely.
    backend = FakeBackend(rules=[(CODE_FENCE, "import torch\n")],
                           default="thinking\n" + CODE_FENCE + "\n")
    conf, rc = confs()
    c = cand()
    segment.advance(backend, [c], conf, rc, count, step=0)
    assert c.mode == CODE
    assert c.text == "## Plan\nthinking\n" + CODE_FENCE + "\nimport torch\n"


def test_the_seam_gains_a_newline_only_when_the_code_does_not_bring_one():
    # head="<HEAD>" prefixes BOTH passes' prompts, so a rule keyed on it would fire for
    # the code pass too. Give the plan pass `default=` (its prompt has no fence yet) and
    # key the one rule on the bare CODE_FENCE, which only the code pass's prompt carries.
    backend = FakeBackend(rules=[(CODE_FENCE, "import torch\n")],
                           default="thinking\n" + CODE_FENCE + "\n")
    conf, rc = confs()
    c = cand()
    segment.advance(backend, [c], conf, rc, count, step=0)
    assert CODE_FENCE + "\nimport torch" in c.text
    assert CODE_FENCE + "import torch" not in c.text


def test_a_plan_that_never_reaches_the_fence_stays_in_prose():
    backend = FakeBackend(rules=[("<HEAD>", "still thinking\n")])
    conf, rc = confs()
    c = cand()
    segment.advance(backend, [c], conf, rc, count, step=0)
    assert c.mode == PROSE
    assert c.done and c.stop == segment.EOS_IN_PLAN


def test_a_code_candidate_that_ends_naturally_is_done():
    backend = FakeBackend(rules=[("<HEAD>", "x = 1\n```\n")])
    conf, rc = confs()
    c = cand(text="## Plan\n" + CODE_FENCE + "\n", mode=CODE)
    segment.advance(backend, [c], conf, rc, count, step=0)
    assert c.done and c.stop == segment.EOS


def test_a_candidate_with_no_budget_left_is_done_without_a_call():
    backend = FakeBackend(rules=[("<HEAD>", "more\n")])
    conf, rc = confs()
    rc.max_new_tokens = 1
    c = cand(text="x" * 400, mode=CODE, spent=999)
    segment.advance(backend, [c], conf, rc, count, step=0)
    assert c.done and c.stop == segment.BUDGET
    assert backend.batches == []


def test_spent_is_recounted_from_the_text_not_accumulated():
    backend = FakeBackend(rules=[("<HEAD>", "x = 1\n")])
    conf, rc = confs()
    c = cand(text="## Plan\n" + CODE_FENCE + "\n", mode=CODE, spent=999)
    segment.advance(backend, [c], conf, rc, count, step=0)
    assert c.spent == count(c.text)


def test_every_candidate_of_every_problem_goes_into_one_call_per_pass():
    """The one-batch property engine.py exists to protect, held here too."""
    backend = FakeBackend(rules=[("<HEAD>", "x = 1\n")], default="x = 1\n")
    conf, rc = confs()
    cands = [cand(text="## Plan\n" + CODE_FENCE + "\n", mode=CODE) for _ in range(12)]
    segment.advance(backend, cands, conf, rc, count, step=0)
    assert len(backend.batches) == 1
    assert len(backend.batches[0]) == 12


def test_calls_bucket_on_exact_room_so_a_short_budget_does_not_starve_the_batch():
    """BUDGET_QUANTUM is 1024 and every segment budget is under it -- quantising would put
    all of these in one bucket and hand them the smallest room in the batch."""
    backend = FakeBackend(default="x = 1\n")
    conf, rc = confs(segment_max_tokens=256)
    roomy = cand(text="## Plan\n" + CODE_FENCE + "\n", mode=CODE)
    tight = cand(text="## Plan\n" + CODE_FENCE + "\n", mode=CODE, spent=rc.max_new_tokens - 5)
    segment.advance(backend, [roomy, tight], conf, rc, count, step=0)
    assert len(backend.batches) == 2


def test_the_code_pass_gets_the_room_left_after_the_plan_pass_spent_some():
    """Room is decremented mid-step by the plan pass's own token count, not left at the
    full segment_max_tokens -- FakeBackend ignores max_tokens and the fixtures would not
    otherwise notice an omitted decrement, so this reads the CODE call's kwargs directly."""
    plan_body = "thinking\n"
    backend = Recorder(rules=[(CODE_FENCE, "codetext\n")],
                        default=plan_body + CODE_FENCE + "\n")
    conf, rc = confs(segment_max_tokens=256)
    c = cand()
    segment.advance(backend, [c], conf, rc, count, step=0)
    # emitted == plan_body + CODE_FENCE: the truncated plan text plus the stop string
    # vLLM (and FakeBackend) keep the tokens for, exactly what the plan call is billed for.
    plan_tokens = count(plan_body + CODE_FENCE)
    assert backend.calls[0]["max_tokens"] == conf.segment_max_tokens
    assert backend.calls[1]["max_tokens"] == conf.segment_max_tokens - plan_tokens


def test_call_does_not_misalign_completions_across_room_buckets():
    """The worst bug this file can have: one candidate's generated text glued onto
    another. Three distinct rooms force >= 2 buckets, and input order [a, b, cc] with
    rooms [256, 100, 50] means bucket iteration (sorted ascending by room) visits cc,
    then b, then a -- the reverse of input order, so a "results come back in input
    order" bug would pair every candidate with a sibling's text."""
    backend = FakeBackend(rules=[("<A>", "AAA\n"), ("<B>", "BBB\n"), ("<C>", "CCC\n")])
    conf, rc = confs(segment_max_tokens=256)
    body = "## Plan\n" + CODE_FENCE + "\n"
    a = cand(text=body, mode=CODE, head="<A>", spent=0)
    b = cand(text=body, mode=CODE, head="<B>", spent=rc.max_new_tokens - 100)
    cc = cand(text=body, mode=CODE, head="<C>", spent=rc.max_new_tokens - 50)
    segment.advance(backend, [a, b, cc], conf, rc, count, step=0)
    assert "AAA" in a.text and "BBB" not in a.text and "CCC" not in a.text
    assert "BBB" in b.text and "AAA" not in b.text and "CCC" not in b.text
    assert "CCC" in cc.text and "AAA" not in cc.text and "BBB" not in cc.text


def test_the_cuts_policy_truncates_a_live_candidate_to_one_chunk():
    body = "".join(f"x{i} = {i}\n" for i in range(40))
    # "length", not the FakeBackend default "stop": this candidate must still be live
    # after the call, or _to_cut_target skips it as an already-finished kernel.
    backend = Finishing("length", rules=[("<HEAD>", body)])
    conf, rc = confs(advance=CUTS, code_steps_per_chunk=4)
    c = cand(text="## Plan\n" + CODE_FENCE + "\n", mode=CODE)
    segment.advance(backend, [c], conf, rc, count, step=0)
    assert c.reached_target
    # Step 0 keeps exactly the first kept cut of the assembled text -- the ruler counts
    # prose and code cuts alike, so the expectation is computed, not hand-counted.
    full = "## Plan\n" + CODE_FENCE + "\n" + body
    cuts = live_cuts(full, code_steps=4)
    assert c.text == full[: cuts[0].char]
    assert c.text != full


def test_the_cuts_policy_never_truncates_a_finished_kernel():
    body = "".join(f"x{i} = {i}\n" for i in range(40)) + "```\n"
    backend = FakeBackend(rules=[("<HEAD>", body)])
    conf, rc = confs(advance=CUTS, code_steps_per_chunk=4)
    c = cand(text="## Plan\n" + CODE_FENCE + "\n", mode=CODE)
    segment.advance(backend, [c], conf, rc, count, step=0)
    assert c.done
    assert c.text.endswith("```\n")


def test_a_candidate_short_of_the_cut_target_is_marked_not_reached():
    backend = FakeBackend(rules=[("<HEAD>", "x = 1\n")])
    # prose_lines_per_chunk defaults to 1, and the "## Plan" header is itself a kept
    # prose cut at that default -- raise it too, or the header alone satisfies the target
    # regardless of code_steps_per_chunk.
    conf, rc = confs(advance=CUTS, code_steps_per_chunk=40, prose_lines_per_chunk=40)
    c = cand(text="## Plan\n" + CODE_FENCE + "\n", mode=CODE)
    segment.advance(backend, [c], conf, rc, count, step=0)
    assert not c.reached_target


def test_left_is_the_run_budget_without_a_cap_and_the_lower_of_both_with_one():
    _, rollout = confs()
    c = cand(text="x" * 400)
    c.spent = 100
    assert segment._left(c, rollout) == 3996
    c.cap = 120
    assert segment._left(c, rollout) == 20
    c.cap = 50
    assert segment._left(c, rollout) == 0


def test_a_candidate_at_its_cap_is_retired_as_budget():
    search, rollout = confs()
    c = cand(mode=CODE, text="```python\n")
    c.spent, c.cap = 3, 3
    segment.advance(FakeBackend(default="x = 1\n"), [c], search, rollout, count, 0)
    assert c.done and c.stop == segment.BUDGET
