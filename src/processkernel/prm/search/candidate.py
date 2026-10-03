"""One live generation in the beam: its text, where the sampler is, and where it came from."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from processkernel.generation.core.model import Problem
from processkernel.generation.core.sampling import PLAN_PREFIX
from processkernel.prm.data.chunks import CODE, PROSE, live_cuts
from processkernel.prm.train.encoding import scored_text_of

_COUNTER = itertools.count()


def new_cid() -> str:
    return f"c{next(_COUNTER)}"


def pkey(problem: Problem) -> tuple[int, int]:
    """A problem's identity across levels -- problem_id alone repeats between them."""
    return (problem.level, problem.problem_id)


@dataclass
class Candidate:
    """A half-written completion, plus the lineage the tree and the diversity floor need."""

    problem: Problem
    head: str                   # render_chat(system, prompt) -- generation continues from this
    prompt: str                 # the bare prompt -- the PRM scores against this, not `head`
    text: str = ""              # the generated half, assembled as sampling.py assembles it
    spent: int = 0              # generated tokens, recounted each step rather than accumulated
    mode: str = PROSE           # which pass the sampler is in; PROSE = still planning
    done: bool = False
    stop: str = ""              # eos | eos_in_plan | budget | steps
    cid: str = field(default_factory=new_cid)
    parent: str | None = None
    scores: list[float] = field(default_factory=list)
    cut_chars: list[int] = field(default_factory=list)
    reached_target: bool = True  # advance=cuts: did this step reach its cut target
    cap: int | None = None       # a per-request token cap below prm_rollout.max_new_tokens
    group: int = 0               # the independent sub-beam this candidate descends from

    def context(self) -> str:
        return self.head + self.text

    def cut(self, conf) -> int:
        """The deepest legal cut offset into `text`, or its length when scoring raw."""
        if not conf.score_at_cut:
            return len(self.text)
        cuts = live_cuts(
            self.text,
            prose_lines=conf.prose_lines_per_chunk,
            code_steps=conf.code_steps_per_chunk,
        )
        return cuts[-1].char if cuts else 0

    def scored_prefix(self, conf) -> str:
        return scored_text_of(self.prompt, self.text[: self.cut(conf)])

    def child(self) -> "Candidate":
        """A copy that shares this candidate's text and names it as the parent.

        The histories are copied, not shared: siblings diverge from here and a shared list
        would record one sibling's scores on all of them.
        """
        return Candidate(
            problem=self.problem, head=self.head, prompt=self.prompt, text=self.text,
            spent=self.spent, mode=self.mode, cid=new_cid(), parent=self.cid,
            scores=list(self.scores), cut_chars=list(self.cut_chars), cap=self.cap, group=self.group,
        )


def root_of(problem: Problem, backend, system: str, prompt: str, think: bool,
            head: str | None = None, cap: int | None = None) -> Candidate:
    """The empty candidate a problem's beam grows from.

    `text` opens with PLAN_PREFIX under the two-pass sampler because that prefill is part of
    the assembled completion (sampling.py) and so part of what v1 stored as `raw` -- the
    layout the PRM was trained on. Single-pass starts empty and in CODE, which is what
    `_single_pass` labels every token of such a run.

    `head` replaces the rendered (system, prompt) chat when an outside driver owns the
    conversation; `prompt` stays what the PRM scores against either way.
    """
    return Candidate(
        problem=problem,
        head=head if head is not None else backend.render_chat(system, prompt),
        prompt=prompt,
        text=PLAN_PREFIX if think else "",
        mode=PROSE if think else CODE,
        cap=cap,
    )
