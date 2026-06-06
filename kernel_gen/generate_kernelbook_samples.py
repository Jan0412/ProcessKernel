"""Generate kernel solutions for GPUMODE/KernelBook rows (multi-sample).

KernelBook is a flat 18k-row dataset with a single ``train`` split and no levels.
Each row's ``python_code`` is converted on the fly into a KernelBench-style
reference (class ``Model`` + positional ``get_init_inputs``) via
:func:`kernelbook_convert.convert_row`, so the prompt the model sees matches what
the KernelBench evaluator later instantiates.

Selected rows are addressed by their dataset row index, which becomes the
problem-id in the output filename ``level_{PL}_problem_{row_idx}_sample_{s}_kernel.py``
(``PL`` = the pseudo-level you also passed to convert_kernelbook.py / eval).

Example:
    python generate_kernelbook_samples.py \\
        --model Qwen/Qwen3-Coder-30B-A3B \\
        --rows 0-499 \\
        --pseudo-level 5 \\
        --num-samples 8 \\
        --temperature 0.8 \\
        --output-dir ~/KernelBench/runs/Qwen3_kernelbook_level5_triton
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
KERNELBENCH_SRC = os.path.join(SCRIPT_DIR, "KernelBench", "src")
sys.path.insert(0, KERNELBENCH_SRC)

from generate_kernels_samples import extract_code_block, generate_samples, load_model
from kernelbook_convert import ConversionError, convert_row

KERNELBOOK_SPLIT = "train"  # KernelBook ships a single split


def parse_rows(spec: str) -> list[int]:
    """Parse '0-499', '1,5,10', or '23' into a list of ints."""
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Generate KernelBook solutions with vLLM sampling"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--rows",
        default=None,
        help="Row indices: single '23', range '0-499', or comma-list '1,5,10'. Omit for --all.",
    )
    parser.add_argument("--all", action="store_true", help="Run every row in the dataset")
    parser.add_argument(
        "--pseudo-level",
        type=int,
        default=5,
        help="Pseudo-level number used in output filenames + eval (default: 5)",
    )
    parser.add_argument("--backend", default="triton", choices=["cuda", "triton", "hip", "tilelang", "cute", "thunderkittens"])
    parser.add_argument("--option", default="one_shot", choices=["zero_shot", "one_shot", "few_shot"])
    parser.add_argument("--gpu-name", default="A100", help="GPU name for hardware-aware prompt")
    parser.add_argument("--include-hardware", action="store_true", help="Inject GPU hardware context into the prompt")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit quantization via bitsandbytes")
    parser.add_argument("--output-dir", required=True, help="Output directory (created if missing)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip rows whose sample files already exist")
    parser.add_argument("--dataset-name", default="GPUMODE/KernelBook")
    parser.add_argument("--max-src-chars", type=int, default=24000, help="Skip rows whose python_code exceeds this many chars")
    parser.add_argument("--num-samples", type=int, default=8, help="Candidates per row (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument(
        "--think-temperature",
        type=float,
        default=None,
        help="If set, two-pass generation split at the ```python fence (prose at this temp, code at --temperature)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    from datasets import load_dataset

    print(f"Loading dataset {args.dataset_name} split={KERNELBOOK_SPLIT} …")
    dataset = load_dataset(args.dataset_name, split=KERNELBOOK_SPLIT)

    if args.all:
        row_ids = list(range(len(dataset)))
    elif args.rows:
        row_ids = parse_rows(args.rows)
    else:
        raise ValueError("Provide --rows or --all.")

    print(f"Output directory : {out_dir}")
    print(f"Rows to solve    : {len(row_ids)}")
    print(f"Pseudo-level     : {args.pseudo_level}")
    print(f"Num samples      : {args.num_samples}")

    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    llm = load_model(args.model, load_in_4bit=args.load_in_4bit)

    for row_id in row_ids:
        try:
            row = dataset[row_id]
        except IndexError:
            print(f"[WARN] row {row_id} out of range, skipping")
            continue

        module_name = row.get("module_name") or row.get("entry_point") or ""
        python_code = row.get("python_code") or ""

        if len(python_code) > args.max_src_chars:
            print(f"[SKIP size] row {row_id} ({module_name})")
            continue

        if args.skip_existing and all(
            os.path.exists(
                os.path.join(out_dir, f"level_{args.pseudo_level}_problem_{row_id}_sample_{i}_kernel.py")
            )
            for i in range(args.num_samples)
        ):
            print(f"[SKIP] row {row_id} ({module_name})")
            continue

        try:
            ref_arch_src = convert_row(python_code, module_name)
        except ConversionError as e:
            print(f"[SKIP convert] row {row_id} ({module_name}): {e}")
            continue

        print(f"\n[{row_id}] {module_name}")

        prompt = get_prompt_for_backend(
            ref_arch_src=ref_arch_src,
            backend=args.backend,
            option=args.option,
            include_hardware=args.include_hardware,
            gpu_name=args.gpu_name if args.include_hardware else None,
        )

        raws = generate_samples(
            llm, prompt, args.max_new_tokens, args.num_samples, args.temperature, args.think_temperature
        )

        for sample_idx, raw in enumerate(raws):
            code = extract_code_block(raw)
            fname = f"level_{args.pseudo_level}_problem_{row_id}_sample_{sample_idx}_kernel.py"
            with open(os.path.join(out_dir, fname), "w") as f:
                f.write(code)
            print(f"  sample_{sample_idx} → {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
