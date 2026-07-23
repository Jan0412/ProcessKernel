"""Arm A5: generate, lint, repair -- up to N rounds, stopping each sample when it is clean.

    round 0   generate 10 samples per problem (no feedback exists yet)
    round r   for every sample the linter still complains about: show it the findings
              and ask for a corrected file
    stop      as soon as a sample is lint-clean, or after --rounds rounds

A sample that goes clean in round 1 costs nothing for the rest of the run, and each
round batches every still-dirty sample across every problem into one vLLM call.

**Round 0 is the baseline.** It is an unrefined generation -- same prompt, same
sampler, no feedback in existence yet -- and it is written to ``rounds/round_0/`` under
canonical filenames, which makes that directory a run dir in its own right. So the
comparison is one generation job and two evals:

    uv run python -m kernel_gen.arms.lintloop --level 1 --all --rounds 3 --num-samples 10
    uv run python -m autotune.eval_run --run-dir runs/<run>                --level 1  # refined
    uv run python -m autotune.eval_run --run-dir runs/<run>/rounds/round_0 --level 1  # baseline

That is not merely cheaper than generating a separate baseline: it is *paired*. Every
slot has its own round-0 ancestor, so the questions that decide whether the loop is
safe to use -- how many broken samples did it fix, and did it break any that already
worked -- are answerable, which two independent draws of 10 could never be.

Note the linter's own numbers are NOT the result. The loop optimizes them by
construction; "findings went down" is circular. The claim is made on eval_results.json.
``lint_loop.jsonl`` is logged for the mechanism (which checks resist repair, and whether
F1.6 passthrough kernels start appearing once a feedback loop exists, as the check notes
predict), not for the headline.

``--trace`` adds a second output, and changes nothing about the first. Alongside the
kernels it writes ``traces/round_{r}/``: one ``.npz`` per attempt holding the token ids
and the top-20 alternatives the model weighed at every step, and an ``attempts.jsonl``
holding the ``## Plan`` prose, the linter's findings with their line numbers, the
plan/code seam offsets and DeepConf's group-confidence summaries. That is training data
for a process reward model -- which step of a generation went wrong, and whether the
model showed any sign of knowing. It costs no extra generation: the numbers were always
computed and always thrown away at ``backend.py``'s ``complete``.

Examples:
    # KernelBench level 1, the real run
    uv run python -m kernel_gen.arms.lintloop --model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
        --level 1 --all --rounds 3 --num-samples 10

    # The same, capturing PRM training data
    uv run python -m kernel_gen.arms.lintloop --model Qwen/Qwen3.6-27B \\
        --level 1 --all --rounds 3 --num-samples 10 --trace

    # KernelBook, same loop, pseudo-level 5
    uv run python -m kernel_gen.arms.lintloop --model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
        --dataset kernelbook --level 5 --rows 0-499 --rounds 3

    # No GPU: render round 0's prompt and exit
    uv run python -m kernel_gen.arms.lintloop --model x --level 1 --problems 0 --dry-run
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from kernel_gen.core import artifacts, cli
from kernel_gen.core.critics import lint_critic
from kernel_gen.core.engine import run_rounds
from kernel_gen.core.model import Attempt, Problem, Trajectory
from kernel_gen.core.prompts import SYSTEM_PROMPT, build_base_prompt, build_repair_prompt
from kernel_gen.core.sampling import SamplingSpec
from kernel_gen.core.sources import load_problems
from kernel_gen.gen_config import print_generation_summary

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_dataset_args(parser)
    cli.add_model_args(parser)
    cli.add_sampling_args(parser)
    cli.add_prompt_args(parser)

    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Max attempts per sample, INCLUDING round 0 (default: 3). --rounds 1 is a "
             "plain generation with the linter run as a no-op observer.",
    )
    parser.add_argument(
        "--lint-checks",
        default=None,
        help="Comma-separated check ids to enforce, e.g. 'F1.2,F1.4' (default: all). "
             "A comma string, never nargs -- a YAML block list is unreadable to the "
             "config parser downstream. See kernel_gen/core/cli.py.",
    )
    parser.add_argument(
        "--feedback-policy",
        default="severity",
        choices=["severity", "fails-only", "all"],
        help="severity (default): show fails, or warns only when there are no fails. "
             "The other two exist so this choice is ablatable.",
    )
    parser.add_argument("--max-findings", type=int, default=8)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Record per-token model internals to traces/ -- token ids, the top-K "
             "alternatives at each step, the plan prose and the linter's line numbers. "
             "This is PRM training data; it changes nothing about what is generated.",
    )
    parser.add_argument(
        "--trace-topk",
        type=int,
        default=20,
        help="Alternatives kept per token (default: 20, which is also vLLM's "
             "max_logprobs). Costs ~6 bytes per token per alternative on disk.",
    )
    parser.add_argument(
        "--trace-window",
        type=int,
        default=512,
        help="Sliding window for the DeepConf group-confidence summaries (default: "
             "512). DeepConf's own 2048 was tuned on math traces; a plan here is "
             "300-800 tokens and a 2048-wide window would average it away entirely.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip slots already recorded in lint_loop.jsonl. Keyed on the record, not "
             "on the kernel file: a file on disk may be an unfinished trajectory's "
             "last write, and resuming from it would score a dirty intermediate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render round 0's prompt and exit -- no model, no GPU. Diff it against "
             "generate_kernels_samples.py's to prove the sampler has not drifted.",
    )
    return parser


def default_output_dir(args: argparse.Namespace) -> str:
    slug = args.model.split("/")[-1]
    tag = "kb" if args.dataset == "kernelbook" else "level"
    return os.path.join(
        REPO_ROOT, "runs", f"{slug}_{tag}{args.level}_lintloop_{args.backend}"
    )


def lint_log_path(out_dir: str) -> str:
    return os.path.join(out_dir, "lint_loop.jsonl")


def load_done_slots(out_dir: str) -> set[tuple[int, int]]:
    return {
        (record["problem_id"], record["sample_id"])
        for record in artifacts.read_jsonl(lint_log_path(out_dir))
    }


def main() -> None:
    args = build_parser().parse_args()
    cli.resolve_dataset_name(args)

    if args.output_dir is None:
        args.output_dir = default_output_dir(args)
    out_dir = args.output_dir  # created only once we commit to running (see --dry-run)

    problems = load_problems(
        args.dataset,
        dataset_name=args.dataset_name,
        level=args.level,
        spec=args.problems,
        all_rows=args.all,
        max_src_chars=args.max_src_chars,
    )
    if not problems:
        raise SystemExit("No problems selected.")

    slots = [(p, s) for p in problems for s in range(args.num_samples)]
    if args.skip_existing:
        done = load_done_slots(out_dir)
        before = len(slots)
        slots = [(p, s) for p, s in slots if (p.problem_id, s) not in done]
        print(f"--skip-existing: {before - len(slots)} slots already recorded, "
              f"{len(slots)} to go")
        if not slots:
            print("Nothing left to do.")
            return

    config = dict(vars(args))
    config.update(
        run_name=os.path.basename(os.path.normpath(out_dir)),
        num_problems=len(problems),
        num_slots=len(slots),
        arm="lintloop",
        script="kernel_gen/arms/lintloop.py",
    )
    print_generation_summary(
        config,
        keys=["model", "arm", "dataset", "dataset_name", "level", "num_problems",
              "num_slots", "num_samples", "rounds", "feedback_policy", "lint_checks",
              "temperature", "think_temperature", "max_new_tokens", "max_model_len",
              "trace", "trace_topk", "output_dir"],
        title="Lint-feedback loop (A5)",
    )

    # Round-0 prompts are identical to a plain generation's, so this closure is also
    # exactly what the baseline arm would use. Memoized: 10 slots share one prompt.
    prompt_cache: dict[int, str] = {}

    def base_prompt(problem: Problem) -> str:
        if problem.problem_id not in prompt_cache:
            prompt_cache[problem.problem_id] = build_base_prompt(
                problem,
                backend=args.backend,
                option=args.option,
                include_hardware=args.include_hardware,
                gpu_name=args.gpu_name,
            )
        return prompt_cache[problem.problem_id]

    def repair_prompt(problem: Problem, attempt: Attempt) -> str:
        return build_repair_prompt(base_prompt(problem), attempt)

    if args.dry_run:
        print("\n" + "=" * 78)
        print(f"DRY RUN — round-0 prompt for problem {problems[0].problem_id} "
              f"({problems[0].name})")
        print("=" * 78)
        print(f"[system]\n{SYSTEM_PROMPT}\n")
        print(f"[user]\n{base_prompt(problems[0])}")
        return

    os.makedirs(out_dir, exist_ok=True)
    cfg_path = artifacts.write_config(out_dir, config, dataset=args.dataset)
    print(f"Saved config     : {cfg_path}")

    from kernel_gen.core.backend import VLLMBackend

    backend = VLLMBackend(
        args.model,
        load_in_4bit=args.load_in_4bit,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        max_num_seqs=args.max_num_seqs,
        max_logprobs=args.trace_topk,
    )

    spec = SamplingSpec(
        system=SYSTEM_PROMPT,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        think_temperature=args.think_temperature if args.think_temperature > 0 else None,
        trace_topk=args.trace_topk if args.trace else None,
    )
    if args.trace:
        # The run-level facts a reader needs and cannot recover from the arrays. Written
        # before the first round, so a crashed run's partial traces are still readable.
        cfg = artifacts.write_trace_config(
            out_dir,
            {
                "model": args.model,
                "logprobs_mode": "raw_logprobs",
                "trace_topk": args.trace_topk,
                "trace_window": args.trace_window,
                "vocab_size": getattr(backend, "vocab_size", None),
                "temperature": args.temperature,
                "think_temperature": args.think_temperature,
            },
        )
        print(f"Saved trace cfg  : {cfg}")
    critic = lint_critic(
        only=set(args.lint_checks.split(",")) if args.lint_checks else None,
        policy=args.feedback_policy,
        max_findings=args.max_findings,
    )

    # A slot is journaled the round it goes done, not at the end of the run: a crash
    # then costs only the slots still in flight, and --skip-existing reads these records
    # back. Journaling on done (never mid-trajectory) is what keeps that flag's
    # invariant -- final() is settled, so no dirty intermediate can be resumed from.
    #
    # Writing the flat kernel here too is not redundant with the write_kernels below:
    # that one only ever sees the slots THIS session ran, so a resumed run would
    # otherwise ship a run dir missing every slot it skipped.
    def checkpoint(round_index: int, active: list[Trajectory]) -> None:
        artifacts.write_attempts(out_dir, active, round_index)
        finished = [t for t in active if t.done]
        artifacts.write_kernels(out_dir, finished)
        artifacts.append_jsonl(lint_log_path(out_dir), [t.to_dict() for t in finished])
        if args.trace:
            # Every slot that ran this round, not only the finished ones: a slot's
            # round-1 attempt is training data whether or not round 2 improved on it,
            # and it is unreachable once the trajectory moves on. Journaled per round
            # for the same reason the kernels are -- a crash costs only what is
            # in flight.
            artifacts.write_traces(
                out_dir,
                active,
                round_index,
                window=args.trace_window,
                vocab_size=getattr(backend, "vocab_size", None),
            )

    trajectories = run_rounds(
        backend,
        slots,
        base_prompt,
        repair_prompt,
        spec,
        critic=critic,
        rounds=args.rounds,
        on_round_end=checkpoint,
    )

    n_written = artifacts.write_kernels(out_dir, trajectories)
    report(trajectories, n_written, args.rounds, out_dir)


def report(
    trajectories: list[Trajectory], n_written: int, rounds: int, out_dir: str
) -> None:
    print("\n" + "=" * 60)
    print(f"  Wrote {n_written} kernels to {out_dir}")
    print("-" * 60)

    clean = [t for t in trajectories if t.final() and t.final().review and t.final().review.clean]
    print(f"  lint-clean at the end : {len(clean)}/{len(trajectories)}")
    for r in range(rounds):
        stopped = [t for t in clean if t.final().round == r]
        ran = [t for t in trajectories if len(t.attempts) > r]
        print(f"  round {r}: {len(ran):>5} slots ran, {len(stopped):>5} went clean here")

    print("-" * 60)
    print("  Next, and this is where the actual result lives:")
    print(f"    uv run python -m autotune.eval_run --run-dir {out_dir}")
    print(f"    uv run python -m autotune.eval_run --run-dir {out_dir}/rounds/round_0")
    print("=" * 60)


if __name__ == "__main__":
    main()
