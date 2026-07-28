"""Argparse groups shared by the arms.

One rule governs every flag added here: **an argparse dest name is a public YAML key.**
The config is persisted as ``dict(vars(args))`` and read back by the flat scanner in
``triton_lint/runs.py``, so a flag must serialize to a single scalar. In particular
``nargs="+"`` is forbidden -- PyYAML writes it as a block list, whose lines all start
with ``-``, which that scanner drops. A multi-value flag is a comma-separated string,
as ``triton_lint``'s own ``--checks`` already is.
"""

from __future__ import annotations

import argparse

DATASET_DEFAULTS = {
    "kernelbench": "ScalingIntelligence/KernelBench",
    "kernelbook": "GPUMODE/KernelBook",
}


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset", default="kernelbench", choices=["kernelbench", "kernelbook"]
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=f"HF dataset id (default: by --dataset, {DATASET_DEFAULTS})",
    )
    # --level ONLY, for both datasets: pass 5 or 6 for KernelBook. An arm carrying both
    # --level and --pseudo-level would write both keys, and runs.py resolves them as
    # `pseudo_level or level` -- so argparse's pseudo-level default would silently
    # override a KernelBench run's real level. See artifacts.write_config.
    parser.add_argument(
        "--level",
        type=int,
        required=True,
        help="KernelBench level (1-3), or the KernelBook pseudo-level (5/6). Goes into "
             "every output filename; must match what eval was staged with.",
    )
    parser.add_argument(
        "--problems",
        "--rows",  # KernelBook is addressed by row index; same dest, same meaning
        default=None,
        help="Which problems to run: '23', '1-49' or '1,5,10'. UNDER --ref-dir these are "
             "problem ids (the staged filename's prefix). WITHOUT it they are dataset "
             "indices -- the KernelBench split is ordered lexicographically, so index 1 "
             "is problem 10. Omit for --all.",
    )
    parser.add_argument("--all", action="store_true", help="Every problem/row in the split")
    # NOT a third --dataset choice. --dataset also decides whether the config records
    # `level` or `pseudo_level` (artifacts.write_config), and a staged dir cannot answer
    # that -- KernelBench/level6 is a pseudo-level, KernelBench/level1 is not. So the
    # reference source is its own flag and --dataset keeps meaning what it means.
    parser.add_argument(
        "--ref-dir",
        default=None,
        help="Read references from a staged level dir (e.g. KernelBench/level6) instead "
             "of converting dataset rows in-process. This is what eval scores, so it "
             "makes the prompt, the linter's shapes and the eval reference the same "
             "bytes; converting again here reproduces the row UNSCALED and silently "
             "disagrees with a dir staged by convert_kernelbook.py --scale.",
    )
    parser.add_argument(
        "--max-src-chars",
        type=int,
        default=24000,
        help="Skip references longer than this. Applied to KernelBook's raw python_code "
             "before conversion, or to the staged file itself under --ref-dir",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HuggingFace model id")
    parser.add_argument("--load-in-4bit", action="store_true", help="bitsandbytes 4-bit")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=40960,
        help="vLLM context length. A repair prompt carries the base prompt AND the "
             "previous kernel AND the findings, so it needs materially more than a "
             "round-0 prompt; 40960 fits the longest one-shot prompt (~16.4k tokens) "
             "plus a full generation (default: 40960).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
        help="Concurrent sequences vLLM may process. Caps CUDA-graph capture and "
             "sampler warmup cost, not throughput of the queue (default: 32).",
    )
    parser.add_argument("--trust-remote-code", action="store_true")


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-samples", type=int, default=10, help="Sample slots per problem")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for the code")
    parser.add_argument(
        "--think-temperature",
        type=float,
        default=1.0,
        help="If set, two-pass generation split at the ```python fence: prose at this "
             "temperature, code at --temperature. Pass 0 to disable.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16384)


def add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default="triton",
        choices=["cuda", "triton", "hip", "tilelang", "cute", "thunderkittens"],
    )
    parser.add_argument(
        "--option", default="one_shot", choices=["zero_shot", "one_shot", "few_shot"]
    )
    parser.add_argument("--gpu-name", default="H100")
    parser.add_argument("--include-hardware", action="store_true")


def resolve_dataset_name(args: argparse.Namespace) -> str:
    """Fill in ``--dataset-name`` from ``--dataset`` before the config is written."""
    if args.dataset_name is None:
        args.dataset_name = DATASET_DEFAULTS[args.dataset]
    return args.dataset_name
