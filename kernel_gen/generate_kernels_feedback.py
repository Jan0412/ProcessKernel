"""
Round-2 kernel generation with execution feedback (experiment arms A3 and A4).

Re-prompts the model with a kernel it already wrote, plus feedback on how that kernel
performed, and asks for a better one. The two arms differ in exactly one thing -- the
feedback text:

    --arm timing   (A3)  "your kernel was correct and ran at X ms vs Y ms baseline"
    --arm tuning   (A4)  the full launch-config sweep: config-to-latency table, which
                         configs broke correctness, whether the winner sat at the grid
                         edge, and an instruction not to hardcode the constants

Both arms are seeded with the *same* kernel (the A2 champion for that problem) and given
the same sampling budget, so any difference between them is attributable to the feedback.
A `seeds.json` is written per run and the two arms' seed hashes must match -- check with
--assert-seeds-match.

Example:
    python generate_kernels_feedback.py \\
        --model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
        --level 1 \\
        --round1-dir runs/Qwen3-Coder-30B-A3B-Instruct_level1_r1_triton \\
        --arm tuning \\
        --num-samples 10 --temperature 0.3 --think-temperature 1.0 \\
        --max-new-tokens 16384 --max-model-len 32768
"""

import argparse
import hashlib
import json
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from gen_config import print_generation_summary, write_generation_config
from generate_kernels_samples import (
    extract_code_block,
    generate_samples,
    load_model,
    parse_problems,
    problem_id_from_name,
)

from autotune.feedback import build_feedback, select_seed


def load_baseline(path: str, level: int) -> dict[int, float]:
    """{problem_id: mean_ms} for one level, from a KernelBench baseline timing JSON."""
    raw = json.loads(open(path).read())
    problems = raw.get(f"level{level}", {})
    out = {}
    for fname, stats in problems.items():
        match = re.match(r"(\d+)_", str(fname))
        if match and isinstance(stats, dict) and stats.get("mean") is not None:
            out[int(match.group(1))] = float(stats["mean"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--level", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--arm", required=True, choices=["timing", "tuning"],
                        help="timing = A3 (control), tuning = A4 (the proposal)")
    parser.add_argument("--round1-dir", required=True,
                        help="run dir of round 1, containing eval_results.json and sweep/")
    parser.add_argument("--sweep-dir", default=None, help="default: <round1-dir>/sweep")
    parser.add_argument("--baseline-file", default=None,
                        help="default: timing/H100/baseline_time_torch.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--problems", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--backend", default="triton",
                        choices=["cuda", "triton", "hip", "tilelang", "cute", "thunderkittens"])
    parser.add_argument("--option", default="one_shot",
                        choices=["zero_shot", "one_shot", "few_shot"])
    parser.add_argument("--gpu-name", default="H100")
    parser.add_argument("--include-hardware", action="store_true")
    parser.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--think-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    # The round-2 prompt carries the seed kernel and (for A4) the config table on top of the
    # round-1 prompt, so it needs materially more context than round 1's 16k default.
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="render the prompts and exit -- no model, no GPU")
    parser.add_argument("--assert-seeds-match", default=None,
                        help="path to the other arm's seeds.json; fails if the seeds differ")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = os.path.join(
            REPO_ROOT, "runs", f"{model_slug}_level{args.level}_r2{args.arm}_triton"
        )
    if args.sweep_dir is None:
        args.sweep_dir = os.path.join(args.round1_dir, "sweep")
    if args.baseline_file is None:
        args.baseline_file = os.path.join(REPO_ROOT, "timing", "H100", "baseline_time_torch.json")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    sweep_summary = json.loads(open(os.path.join(args.sweep_dir, "sweep_summary.json")).read())
    baseline = load_baseline(args.baseline_file, args.level)

    from datasets import load_dataset

    dataset_split = f"level_{args.level}"
    print(f"Loading dataset {args.dataset_name} split={dataset_split} …")
    dataset = load_dataset(args.dataset_name, split=dataset_split)

    if args.all:
        problem_ids = list(range(0, len(dataset)))
    elif args.problems:
        problem_ids = parse_problems(args.problems)
    else:
        raise ValueError("Provide --problems or --all.")

    # Build every prompt first. This is where a problem can drop out of the experiment, and
    # it must drop out of BOTH arms identically -- so the decision is made from the round-1
    # sweep alone, never from anything arm-specific.
    prompts, seeds, dropped = {}, {}, {}
    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    for problem_id in problem_ids:
        try:
            problem = dataset[problem_id]
        except IndexError:
            dropped[problem_id] = "not in dataset"
            continue
        ref_arch_src = problem["code"]
        problem_name = problem.get("name", f"problem_{problem_id:04d}.py")
        real_id = problem_id_from_name(problem_name, problem_id)

        seed = select_seed(sweep_summary, real_id)
        if seed is None:
            dropped[real_id] = "no correct round-1 sample was swept"
            continue
        if real_id not in baseline:
            dropped[real_id] = "no baseline timing for this problem"
            continue

        seed_path = os.path.join(args.round1_dir, f"{seed['kernel']}.py")
        seed_src = open(seed_path).read()

        base_prompt = get_prompt_for_backend(
            ref_arch_src=ref_arch_src,
            backend=args.backend,
            option=args.option,
            include_hardware=args.include_hardware,
            gpu_name=args.gpu_name if args.include_hardware else None,
        )
        feedback = build_feedback(args.arm, seed, baseline[real_id])
        prompts[real_id] = (
            f"{base_prompt}\n\n"
            f"## Your previous solution\n\n"
            f"```python\n{seed_src}\n```\n\n"
            f"{feedback}"
        )
        seeds[str(real_id)] = {
            "problem_name": problem_name,
            "seed_kernel": seed["kernel"],
            "seed_sha256": hashlib.sha256(seed_src.encode()).hexdigest(),
            "identity_ms": seed.get("identity_ms"),
            "best_ms": seed.get("best_ms"),
            "best_config": seed.get("best_config"),
            "tuning_gain": seed.get("tuning_gain"),
            "baseline_ms": baseline[real_id],
        }

    with open(os.path.join(out_dir, "seeds.json"), "w") as f:
        json.dump({"arm": args.arm, "dropped": dropped, "seeds": seeds}, f, indent=2)

    if args.assert_seeds_match:
        other = json.loads(open(args.assert_seeds_match).read())
        mine = {k: v["seed_sha256"] for k, v in seeds.items()}
        theirs = {k: v["seed_sha256"] for k, v in other["seeds"].items()}
        if mine != theirs:
            diff = sorted(set(mine) ^ set(theirs)) or [
                k for k in mine if mine[k] != theirs.get(k)
            ]
            raise SystemExit(
                f"FAIRNESS VIOLATION: the two arms were seeded with different kernels "
                f"({len(diff)} problems differ, e.g. {diff[:5]}). A4-vs-A3 would be "
                f"uninterpretable. Regenerate both arms from the same sweep_summary.json."
            )
        print(f"seed check OK: {len(mine)} problems, identical seeds in both arms")

    config = dict(vars(args))
    config.update(
        run_name=os.path.basename(os.path.normpath(out_dir)),
        dataset_split=dataset_split,
        num_problems=len(prompts),
        num_dropped=len(dropped),
        script=os.path.basename(__file__),
    )
    print_generation_summary(
        config,
        keys=["model", "arm", "level", "round1_dir", "num_problems", "num_dropped",
              "num_samples", "temperature", "think_temperature", "backend", "option",
              "max_new_tokens", "max_model_len", "output_dir"],
        title=f"KernelBench round-2 generation (arm={args.arm})",
    )
    cfg_path = write_generation_config(out_dir, config)
    print(f"Saved config     : {cfg_path}")
    print(f"Saved seeds      : {os.path.join(out_dir, 'seeds.json')}")
    if dropped:
        print(f"Dropped {len(dropped)} problems (no correct round-1 kernel or no baseline)")

    if args.dry_run:
        example = next(iter(prompts))
        print("\n" + "=" * 78)
        print(f"DRY RUN — prompt for problem {example} (arm={args.arm}), "
              f"{len(prompts[example]):,} chars")
        print("=" * 78)
        print(prompts[example])
        return

    llm = load_model(
        args.model,
        load_in_4bit=args.load_in_4bit,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    for i, (real_id, prompt) in enumerate(sorted(prompts.items())):
        if args.skip_existing and all(
            os.path.exists(os.path.join(
                out_dir, f"level_{args.level}_problem_{real_id}_sample_{s}_kernel.py"))
            for s in range(args.num_samples)
        ):
            print(f"[SKIP] problem {real_id}")
            continue

        print(f"\n[{i + 1}/{len(prompts)}] problem {real_id} "
              f"(seed {seeds[str(real_id)]['seed_kernel']}, "
              f"gain {seeds[str(real_id)]['tuning_gain']})")

        raws = generate_samples(
            llm, prompt, args.max_new_tokens, args.num_samples,
            args.temperature, args.think_temperature,
        )
        for sample_idx, raw in enumerate(raws):
            code = extract_code_block(raw)
            fname = f"level_{args.level}_problem_{real_id}_sample_{sample_idx}_kernel.py"
            with open(os.path.join(out_dir, fname), "w") as f:
                f.write(code)
            print(f"  sample_{sample_idx} → {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
