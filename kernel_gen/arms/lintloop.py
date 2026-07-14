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

Examples:
    # KernelBench level 1, the real run
    uv run python -m kernel_gen.arms.lintloop --model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
        --level 1 --all --rounds 3 --num-samples 10

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


def load_done_slots(out_dir: str) -> set[tuple[int, int]]:
    return {
        (record["problem_id"], record["sample_id"])
        for record in artifacts.read_jsonl(os.path.join(out_dir, "lint_loop.jsonl"))
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
              "output_dir"],
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
    )

    spec = SamplingSpec(
        system=SYSTEM_PROMPT,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        think_temperature=args.think_temperature if args.think_temperature > 0 else None,
    )
    critic = lint_critic(
        only=set(args.lint_checks.split(",")) if args.lint_checks else None,
        policy=args.feedback_policy,
        max_findings=args.max_findings,
    )

    trajectories = run_rounds(
        backend,
        slots,
        base_prompt,
        repair_prompt,
        spec,
        critic=critic,
        rounds=args.rounds,
        on_round_end=lambda r, active: artifacts.write_attempts(out_dir, active, r),
    )

    n_written = artifacts.write_kernels(out_dir, trajectories)
    artifacts.append_jsonl(
        os.path.join(out_dir, "lint_loop.jsonl"),
        [t.to_dict() for t in trajectories],
    )
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
