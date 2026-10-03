"""Generate kernels: one completion per sample slot, KernelBench's one-shot Triton prompt.

Writes the kernels flat in the output dir (what eval scores and the PRM and ORM data
builders read), ``generation_config.yaml`` and ``generation.jsonl``, the journal
``--skip-existing`` resumes from.

``--trace`` adds ``traces/``: one ``.npz`` per slot holding the token ids and the
top-K alternatives the model weighed at every step, and an ``attempts.jsonl`` with the full
prompt, the raw completion including the ``## Plan`` prose, the extracted code and
DeepConf's group-confidence summaries. That is the PRM's training data, at no extra
generation cost.

Examples:
    # KernelBench level 1, 4 samples per problem
    uv run python -m processkernel.generation.generate --model Qwen/Qwen3.6-35B-A3B \\
        --level 1 --all --num-samples 4

    # KernelBook, pseudo-level 6. --ref-dir is not optional in practice: it points the
    # prompt at the same staged files eval scores. Without it the row is re-converted
    # here UNSCALED, and the model is asked about a 4x4 problem it will be graded on at
    # 2048x2048. See core/sources.py.
    uv run python -m processkernel.generation.generate --model deepseek-ai/DeepSeek-V4-Flash \\
        --dataset kernelbook --level 6 --ref-dir KernelBench/KernelBench/level6 --rows 0-499 --trace

    # No GPU: render the prompt and exit
    uv run python -m processkernel.generation.generate --model x --level 1 --problems 0 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from processkernel.checker.core.naming import staged_kernel_filename
from processkernel.generation.core import artifacts, cli
from processkernel.generation.core.engine import generate
from processkernel.generation.core.model import Problem, Trajectory
from processkernel.generation.core.prompt_deltas import DELTA_ORDER, parse_deltas
from processkernel.generation.core.prompts import BACKEND, OPTION, SYSTEM_PROMPT, build_base_prompt
from processkernel.generation.core.sampling import SamplingSpec
from processkernel.generation.core.sources import load_problems
from processkernel.generation.gen_config import print_generation_summary

# this file is src/processkernel/<package>/<module>.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JOURNAL = "generation.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_dataset_args(parser)
    cli.add_model_args(parser)
    cli.add_sampling_args(parser)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Record per-token model internals to traces/ -- token ids, the top-K "
             "alternatives at each step and the plan prose. This is PRM training data; "
             "it changes nothing about what is generated.",
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
    parser.add_argument(
        "--prompt-deltas",
        default="",
        help=f"Comma-separated blocks appended to the prompt: {', '.join(DELTA_ORDER)}. "
             "The paper's PRM training runs used both; empty (default) is the paper's "
             "prompt unchanged.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=f"Skip slots already recorded in {JOURNAL}. Keyed on the journal, not on "
             "the kernel file: a slot is journaled only after its kernel and trace are "
             "on disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the prompt of the first problem and exit -- no model, no GPU.",
    )
    return parser


def default_output_dir(args: argparse.Namespace) -> str:
    slug = args.model.split("/")[-1]
    tag = "kb" if args.dataset == "kernelbook" else "level"
    return os.path.join(REPO_ROOT, "runs", f"{slug}_{tag}{args.level}_{BACKEND}")


def journal_path(out_dir: str) -> str:
    return os.path.join(out_dir, JOURNAL)


def load_done_slots(out_dir: str) -> set[tuple[int, int]]:
    return {
        (record["problem_id"], record["sample_id"])
        for record in artifacts.read_jsonl(journal_path(out_dir))
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cli.resolve_dataset_name(args)

    # Both knobs open the assistant turn: --enable-thinking leaves the model's <think>
    # block open, a non-zero --think-temperature prefills "## Plan" into it. Together
    # the plan and the code are written inside a block the model never closes, which
    # yields scratch work that still parses -- so it fails here rather than after a
    # two-day run.
    if args.enable_thinking and args.think_temperature:
        raise SystemExit(
            "--enable-thinking with a non-zero --think-temperature writes the plan "
            "and the code inside the model's own <think> block, which it never "
            "closes. Use --think-temperature 0 with --enable-thinking, or drop "
            "the flag."
        )

    try:
        deltas = parse_deltas(args.prompt_deltas)
    except ValueError as e:
        raise SystemExit(f"--prompt-deltas: {e}") from None

    if args.output_dir is None:
        args.output_dir = default_output_dir(args)
    out_dir = args.output_dir  # created only once we commit to running (see --dry-run)

    problems = load_problems(
        args.dataset,
        ref_dir=args.ref_dir,
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
        # Shard-qualified: runs.py stamps this onto every SampleRef, and four shards all
        # called "shard_0N" would be indistinguishable once pooled.
        run_name=artifacts.eval_run_name(out_dir),
        num_problems=len(problems),
        num_slots=len(slots),
        backend=BACKEND,
        option=OPTION,
        arm="generate",
        script="src/processkernel/generation/generate.py",
    )
    print_generation_summary(
        config,
        keys=["model", "dataset", "dataset_name", "ref_dir", "level", "num_problems",
              "num_slots", "num_samples", "temperature", "top_p", "top_k",
              "think_temperature", "max_new_tokens", "max_model_len", "trace",
              "trace_topk", "prompt_deltas", "output_dir"],
        title="Generation",
    )

    # Memoized: every sample of a problem shares one prompt.
    prompt_cache: dict[int, str] = {}

    def prompt_for(problem: Problem) -> str:
        if problem.problem_id not in prompt_cache:
            prompt_cache[problem.problem_id] = build_base_prompt(problem, deltas)
        return prompt_cache[problem.problem_id]

    if args.dry_run:
        print("\n" + "=" * 78)
        print(f"DRY RUN -- prompt for problem {problems[0].problem_id} ({problems[0].name})")
        print("=" * 78)
        print(f"[system]\n{SYSTEM_PROMPT}\n")
        print(f"[user]\n{prompt_for(problems[0])}")
        return

    os.makedirs(out_dir, exist_ok=True)
    cfg_path = artifacts.write_config(out_dir, config, dataset=args.dataset)
    print(f"Saved config     : {cfg_path}")

    from processkernel.generation.core.backend import VLLMBackend

    backend = VLLMBackend(
        args.model,
        load_in_4bit=args.load_in_4bit,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        max_num_seqs=args.max_num_seqs,
        max_logprobs=args.trace_topk,
        enable_thinking=args.enable_thinking,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    spec = SamplingSpec(
        system=SYSTEM_PROMPT,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        think_temperature=args.think_temperature if args.think_temperature > 0 else None,
        trace_topk=args.trace_topk if args.trace else None,
    )
    if args.trace:
        # A slot in flight when a run died was never journaled, so it runs again here; its
        # old trace record would else point at the new arrays (contract 4).
        n_pruned = artifacts.prune_traces(
            out_dir,
            {
                staged_kernel_filename(p.level, p.problem_id, s)[: -len(".py")]
                for p, s in slots
            },
        )
        if n_pruned:
            print(f"Pruned {n_pruned} trace records for slots this session regenerates")

        # The run-level facts a reader needs and cannot recover from the arrays.
        cfg = artifacts.write_trace_config(
            out_dir,
            {
                "model": args.model,
                "logprobs_mode": "raw_logprobs",
                "trace_topk": args.trace_topk,
                "trace_window": args.trace_window,
                "vocab_size": getattr(backend, "vocab_size", None),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "think_temperature": args.think_temperature,
            },
        )
        print(f"Saved trace cfg  : {cfg}")

    trajectories = generate(backend, slots, prompt_for, spec)
    n_written = write_outputs(
        out_dir,
        trajectories,
        trace=args.trace,
        window=args.trace_window,
        vocab_size=getattr(backend, "vocab_size", None),
    )
    report(n_written, out_dir, args.level, args.num_samples)


def write_outputs(
    out_dir: str,
    trajectories: list[Trajectory],
    *,
    trace: bool,
    window: int,
    vocab_size: int | None,
) -> int:
    """Kernels, traces, then the journal -- in that order.

    The journal goes last so a crash mid-write leaves the slot NOT marked done, and
    ``--skip-existing`` regenerates it rather than trusting a half-written slot.
    """
    n_written = artifacts.write_kernels(out_dir, trajectories)
    if trace:
        artifacts.write_traces(
            out_dir,
            trajectories,
            window=window,
            vocab_size=vocab_size,
            system_prompt=SYSTEM_PROMPT,
        )
    artifacts.append_jsonl(journal_path(out_dir), [t.to_dict() for t in trajectories])
    return n_written


def report(n_written: int, out_dir: str, level: int, num_samples: int) -> None:
    name = artifacts.eval_run_name(out_dir)
    # eval reads runs_dir/<name>, and name may span several dirs (a shard)
    runs_dir = os.path.abspath(out_dir)
    for _ in name.split("/"):
        runs_dir = os.path.dirname(runs_dir)
    print("\n" + "=" * 60)
    print(f"  Wrote {n_written} kernels to {out_dir}")
    print("-" * 60)
    print("  Next, grade them with the patched KernelBench (kernelbench.patch),")
    print("  from inside the KernelBench checkout:")
    print(f"    uv run python scripts/eval_from_generations.py run_name={shlex.quote(name)} "
          f"runs_dir={shlex.quote(runs_dir)} dataset_src=local level={level} "
          f"backend=triton num_samples_per_problem={num_samples}")
    print("=" * 60)


if __name__ == "__main__":
    main()
