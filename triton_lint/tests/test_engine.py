"""The refinement loop's two non-negotiable properties: it batches, and it degrades.

Round-major batching is the reason this engine exists, and it is invisible in the
output -- a per-problem loop produces byte-identical kernels, just far too slowly to
run. So it is pinned here: a regression to "loop over the problems" must FAIL a test,
not merely be slow.

Degrading is the other half. A run is hours of GPU time; one unparseable completion or
one crashing critic must cost that slot, not the run.
"""

from __future__ import annotations

from kernel_gen.core.backend import FakeBackend
from kernel_gen.core.engine import run_rounds
from kernel_gen.core.model import Problem, Review
from kernel_gen.core.sampling import SamplingSpec

SPEC = SamplingSpec(think_temperature=None)  # single pass: one backend call per round

BAD = "```python\nbad\n```"
GOOD = "```python\ngood\n```"

REPAIR_MARKER = "## Your previous solution"


def problems(n: int) -> list[Problem]:
    return [
        Problem(level=1, problem_id=i, name=f"{i}_P.py", ref_arch_src="ref") for i in range(n)
    ]


def slots(probs: list[Problem], num_samples: int) -> list[tuple[Problem, int]]:
    return [(p, s) for p in probs for s in range(num_samples)]


def build_prompt(problem: Problem) -> str:
    return f"solve problem {problem.problem_id}"


def build_repair_prompt(problem: Problem, attempt) -> str:
    return f"solve problem {problem.problem_id}\n{REPAIR_MARKER}\n{attempt.code}\n{attempt.review.text}"


def critic(problem: Problem, code: str) -> Review:
    """Clean iff the code says so. Stands in for the linter."""
    dirty = "bad" in code
    return Review(
        text="F1.2: you never launched the kernel",
        clean=not dirty,
        data={"n_fail": 1 if dirty else 0, "n_warn": 0, "parse_status": "ok"},
    )


def run(backend, *, rounds=1, critic_fn=None, n_problems=3, num_samples=4, on_round_end=None):
    return run_rounds(
        backend,
        slots(problems(n_problems), num_samples),
        build_prompt,
        build_repair_prompt,
        SPEC,
        critic=critic_fn,
        rounds=rounds,
        on_round_end=on_round_end,
    )


# -- batching --------------------------------------------------------------


def test_one_round_is_exactly_one_batch_across_every_problem():
    # THE test. 3 problems x 4 samples is ONE call of 12 prompts, not 3 calls of 4 and
    # certainly not 12 calls of 1. Everything else here is a detail; this is the design.
    backend = FakeBackend(default=GOOD)
    trajs = run(backend, rounds=1, critic_fn=None, n_problems=3, num_samples=4)

    assert len(backend.batches) == 1
    assert len(backend.batches[0]) == 12
    assert len(trajs) == 12
    assert all(len(t.attempts) == 1 for t in trajs)


def test_the_batch_shrinks_to_only_the_slots_that_still_need_work():
    # Slot (problem 1, sample 0) is the only one that comes out dirty.
    backend = FakeBackend(rules=[(REPAIR_MARKER, GOOD), ("solve problem 1", BAD)], default=GOOD)
    trajs = run(backend, rounds=3, critic_fn=critic, n_problems=3, num_samples=2)

    round_0, round_1 = backend.batches
    assert len(round_0) == 6
    assert len(round_1) == 2  # both slots of problem 1, and nothing else
    assert all("solve problem 1" in p and REPAIR_MARKER in p for p in round_1)

    # Round 2 never happened: everything went clean, so the loop stopped early.
    assert len(backend.batches) == 2


def test_a_clean_slot_stops_while_its_siblings_keep_going():
    # The slot, not the problem, is the unit of work -- samples of the same problem must
    # be able to stop on different rounds.
    backend = FakeBackend(rules=[(REPAIR_MARKER, GOOD)], default=BAD)
    trajs = run(backend, rounds=2, critic_fn=critic, n_problems=1, num_samples=3)

    # Everything is dirty at round 0, so all 3 slots repair, and all 3 come back clean.
    assert [len(b) for b in backend.batches] == [3, 3]
    assert all(len(t.attempts) == 2 for t in trajs)
    assert all(t.final().code == "good" for t in trajs)


def test_no_critic_means_no_second_round_even_when_rounds_is_high():
    # Without feedback a "repair" is just a resample -- a different experiment.
    backend = FakeBackend(default=BAD)
    run(backend, rounds=3, critic_fn=None)

    assert len(backend.batches) == 1


# -- what gets kept --------------------------------------------------------


def test_a_slot_that_never_goes_clean_keeps_its_best_attempt():
    backend = FakeBackend(default=BAD)  # nothing ever repairs
    trajs = run(backend, rounds=3, critic_fn=critic, n_problems=1, num_samples=1)

    traj = trajs[0]
    assert len(traj.attempts) == 3
    assert traj.done  # out of rounds, not clean
    assert traj.final() is not None  # never dropped: N samples per problem is a contract
    assert traj.final().round == 0  # all three tie, so the earliest wins


def test_the_final_kernel_is_the_repaired_one():
    backend = FakeBackend(rules=[(REPAIR_MARKER, GOOD)], default=BAD)
    trajs = run(backend, rounds=2, critic_fn=critic, n_problems=1, num_samples=1)

    assert trajs[0].final().code == "good"
    assert trajs[0].final().round == 1


# -- degrading -------------------------------------------------------------


def test_an_unparseable_completion_degrades_that_slot_only():
    backend = FakeBackend(rules=[("solve problem 0", "")], default=GOOD)
    trajs = run(backend, rounds=1, critic_fn=critic, n_problems=2, num_samples=1)

    assert trajs[0].final().code == ""  # empty, recorded, still written
    assert trajs[1].final().code == "good"  # its sibling is untouched


def test_a_crashing_critic_does_not_abort_the_run():
    def explodes(problem, code):
        raise RuntimeError("the linter tripped over something")

    backend = FakeBackend(default=GOOD)
    trajs = run(backend, rounds=3, critic_fn=explodes, n_problems=2, num_samples=2)

    assert len(trajs) == 4
    assert all(t.attempts for t in trajs)
    assert all(t.final().review is None for t in trajs)
    # No verdict means nothing to say in a repair prompt, so the slot stops rather than
    # being re-prompted with empty feedback.
    assert len(backend.batches) == 1


def test_a_crashing_critic_is_never_mistaken_for_a_clean_one():
    def explodes(problem, code):
        raise RuntimeError("boom")

    trajs = run(FakeBackend(default=BAD), rounds=1, critic_fn=explodes, n_problems=1, num_samples=1)
    assert trajs[0].to_dict()["clean"] is False


# -- the caller's hook -----------------------------------------------------


def test_on_round_end_sees_the_slots_that_ran_that_round():
    seen: list[tuple[int, int]] = []
    backend = FakeBackend(rules=[(REPAIR_MARKER, GOOD), ("solve problem 1", BAD)], default=GOOD)
    run(
        backend,
        rounds=3,
        critic_fn=critic,
        n_problems=3,
        num_samples=2,
        on_round_end=lambda r, active: seen.append((r, len(active))),
    )

    assert seen == [(0, 6), (1, 2)]
