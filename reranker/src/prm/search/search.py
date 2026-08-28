"""The step-major loop: every problem's beam advances together, in one batched pass a step.

Problem-major would be the obvious shape and is the one to refuse. A step here holds
`beam_width x expand` candidates for every problem at once -- at 4x4 over 50 problems that is
800 prompts in one call. Looping the problems would make it 50 calls of 16, which is the same
mistake kernel_gen's engine exists to avoid.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from reranker.src.prm.search import segment, select
from reranker.src.prm.search.candidate import pkey, root_of


@dataclass
class Result:
    pool: dict = field(default_factory=lambda: defaultdict(list))
    pruned: dict = field(default_factory=lambda: defaultdict(list))
    #: one {"step", "rows"} dict per step. Rows are plain summaries, never Candidate
    #: references: a child copies its parent's full text, so keeping every step's objects
    #: alive would hold GBs on a large run -- and run.py wants exactly these rows anyway.
    steps: list = field(default_factory=list)


def search(backend, problems, conf, rollout_conf, count, scorer, encoder, system,
           build_prompt, rng) -> Result:
    """Run the beam to exhaustion or `max_steps`; return every finished candidate."""
    # `two_pass`, not `is not None`: think_temperature is a float, so 0.0 -- the value a
    # native-thinking run records -- passes an `is not None` test and would open every beam
    # with a "## Plan" prefill the source policy never wrote.
    think = rollout_conf.two_pass
    live = {
        pkey(p): [root_of(p, backend, system, build_prompt(p), think)]
        for p in problems
    }
    result = Result()

    for step in range(conf.max_steps):
        n = conf.beam_width if step == 0 else conf.expand
        children = [c.child() for cs in live.values() for c in cs for _ in range(n)]
        if not children:
            break
        segment.advance(backend, children, conf, rollout_conf, count, step)

        alive = []
        for c in children:
            if c.done:
                result.pool[pkey(c.problem)].append(c)
            else:
                alive.append(c)
        scores = select.score(alive, conf, scorer, encoder, rng)

        grouped = defaultdict(list)
        for c, s in zip(alive, scores):
            c.scores.append(s)
            c.cut_chars.append(c.cut(conf))
            grouped[pkey(c.problem)].append((c, s))

        live, kept_ids = {}, set()
        for key, pairs in grouped.items():
            kept, dropped = select.top_b([c for c, _ in pairs], [s for _, s in pairs], conf)
            live[key] = kept
            kept_ids.update(c.cid for c in kept)
            if conf.keep_pruned:
                result.pruned[key].extend(dropped)
        result.steps.append({"step": step, "rows": [_row(c, c.cid in kept_ids) for c in children]})

    # Whatever is still live when the step budget runs out is a truncated kernel, not a lost
    # one: it is written, graded, and marked so the report can separate it from a real EOS.
    for key, cs in live.items():
        for c in cs:
            c.done, c.stop = True, segment.STEPS
            result.pool[key].append(c)
    return result


def _row(c, kept: bool) -> dict:
    """One child's summary for the tree artifact. A done child was not scored this step."""
    return {
        "cid": c.cid, "parent": c.parent, "pkey": pkey(c.problem),
        "score": None if c.done else (c.scores[-1] if c.scores else None),
        "kept": kept, "tokens": c.spent,
        "cut_char": None if c.done else (c.cut_chars[-1] if c.cut_chars else None),
        "reached_target": c.reached_target, "done": c.done, "stop": c.stop,
    }
