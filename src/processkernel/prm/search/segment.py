"""One step of generation: every live candidate grows by one segment, in two calls at most.

Batching is the whole point, exactly as in the generation engine -- a step collects every
candidate of every problem into one call per pass, never one call per problem or per
candidate. `test_segment.py` pins that.
"""

from __future__ import annotations

from collections import defaultdict

from processkernel.generation.core.sampling import CODE_FENCE
from processkernel.config import CUTS
from processkernel.prm.data.chunks import CODE, PROSE, live_cuts
from processkernel.prm.rollout.rollout import budget

EOS, EOS_IN_PLAN, BUDGET, STEPS = "eos", "eos_in_plan", "budget", "steps"


def _left(c, rollout_conf) -> int:
    """Tokens this candidate may still generate: the run's budget, or its own cap if lower."""
    left = budget(c.spent, rollout_conf)
    return left if c.cap is None else min(left, max(0, c.cap - c.spent))


def advance(backend, cands, conf, rollout_conf, count, step: int) -> None:
    """Grow every candidate by one segment, in place."""
    room = {}
    for c in cands:
        room[c.cid] = min(conf.segment_max_tokens, _left(c, rollout_conf))
        if room[c.cid] < 1:
            c.done, c.stop = True, BUDGET

    plans = [c for c in cands if not c.done and c.mode == PROSE]
    for c, out in zip(plans, _call(backend, plans, room, rollout_conf.think_temperature,
                                   [CODE_FENCE])):
        c.text += out.text
        room[c.cid] = max(0, room[c.cid] - len(out.token_ids))
        if out.stop_reason == CODE_FENCE:
            # The bare fence, exactly as sampling.py writes pass 2's prompt. Its newline is
            # decided later, against the code that follows (KGEN-21).
            c.text += CODE_FENCE
            c.mode = CODE
        elif out.finish_reason != "length":
            # The model ended the turn without ever opening a fence. There is no kernel in
            # this candidate and there never will be.
            c.done, c.stop = True, EOS_IN_PLAN

    codes = [c for c in cands if not c.done and c.mode == CODE and room[c.cid] >= 1]
    for c, out in zip(codes, _call(backend, codes, room, rollout_conf.temperature, None)):
        if c.text.endswith(CODE_FENCE) and not out.text.startswith("\n"):
            c.text += "\n"
        c.text += out.text
        if out.finish_reason != "length" and out.stop_reason is None:
            c.done, c.stop = True, EOS

    for c in cands:
        c.spent = count(c.text)
        if not c.done and _left(c, rollout_conf) < 1:
            c.done, c.stop = True, BUDGET
        if conf.advance == CUTS:
            _to_cut_target(c, conf, count, step)


def _to_cut_target(c, conf, count, step: int) -> None:
    """Cut back to this step's chunk, so every candidate is compared at one chunk index.

    One chunk per step: the chunker's `code_steps_per_chunk` / `prose_lines_per_chunk` are
    the stride, so step N truncates at kept-cut index N. A done candidate is never cut: it
    holds a complete kernel and truncating would destroy the closing fence. A candidate that
    did not reach the target keeps everything it has and is counted -- a persistent
    `reached_target=False` rate means segment_max_tokens is too small for the chunk size,
    which is a tuning fact and not a failure.
    """
    target = step
    cuts = live_cuts(c.text, prose_lines=conf.prose_lines_per_chunk,
                     code_steps=conf.code_steps_per_chunk)
    c.reached_target = len(cuts) > target
    if not c.reached_target or c.done:
        return
    c.text = c.text[: cuts[target].char]
    c.spent = count(c.text)


def _call(backend, cands, room, temperature, stop):
    """One backend call per distinct room, realigned to `cands`. Never one call per candidate.

    Bucketed on the exact room, not `room // BUDGET_QUANTUM` as rollout.py does: that quantum
    is 1024 and every segment budget is under it, so quantising collapses the batch into one
    bucket and hands every member the smallest room in it. Exact bucketing is also cheap here
    -- almost every candidate has the full `segment_max_tokens` and only the few near the
    generation cap differ.
    """
    out = [None] * len(cands)
    if not cands:
        return out
    buckets = defaultdict(list)
    for i, c in enumerate(cands):
        buckets[room[c.cid]].append(i)
    for size, idx in sorted(buckets.items()):
        completions = backend.complete_traced(
            [cands[i].context() for i in idx],
            temperature=temperature,
            max_tokens=size,
            stop=stop,
        )
        for i, completion in zip(idx, completions):
            out[i] = completion
    return out
