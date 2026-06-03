"""
Generate kernel solutions for KernelBench problems using HuggingFace beam search.

Each beam candidate is saved as a separate file with a _vN suffix.
Outputs are written to {model_name_slug}_generated_kernels_beam/level_{N}_beam/ next to this script.

Example (SLURM):
    python generate_kernels_beam.py \\
        --model facebook/KernelLLM \\
        --level 1 \\
        --problems 0-49 \\
        --num-beams 4 \\
        --backend triton \\
        --option one_shot \\
        --gpu-name A100

Example (single problem):
    python generate_kernels_beam.py --model facebook/KernelLLM --level 1 --problems 23 --num-beams 4
"""

import argparse
import gc
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELBENCH_SRC = os.path.join(SCRIPT_DIR, "KernelBench", "src")
sys.path.insert(0, KERNELBENCH_SRC)


def parse_problems(spec: str) -> list[int]:
    """Parse '1-49', '1,5,10', or '23' into a list of ints."""
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def extract_code_block(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip())


def load_model(model_id: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading tokenizer for {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    kwargs = dict(device_map="auto", dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading model {model_id} …")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    import torch as _t
    vram_gb = _t.cuda.memory_allocated() / 1e9
    print(f"Model loaded on {next(model.parameters()).device} | VRAM used: {vram_gb:.1f} GB")
    return model, tokenizer


def generate_kernel_beam(
    model, tokenizer, prompt: str, max_new_tokens: int, num_samples: int, temperature: float
) -> list[str]:
    import torch

    SYSTEM_MSG = (
        "You write custom kernels to replace the pytorch operators in the given "
        "architecture to get speedups.\n\n"
        "You have complete freedom to choose the set of operators you want to replace. "
        "You may replace some operators with custom kernels and leave others unchanged.\n\n"
        "You need to provide the complete Python code wrapped in a Python code block "
        "that starts with ```python and ends with ```."
    )

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]

    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda:0")

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    prompt_len = input_ids.shape[-1]

    results = []
    for _ in range(num_samples):
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        results.append(
            tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
        )
        del output_ids
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate KernelBench solutions with HuggingFace beam search"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID, e.g. facebook/KernelLLM")
    parser.add_argument("--level", type=int, required=True, choices=[1, 2, 3], help="KernelBench level (1/2/3)")
    parser.add_argument(
        "--problems",
        default=None,
        help="Problem IDs: single '23', range '1-49', or comma-list '1,5,10'. Omit to use --all.",
    )
    parser.add_argument("--all", action="store_true", help="Run all problems in the level")
    parser.add_argument("--backend", default="triton", choices=["cuda", "triton", "hip", "tilelang", "cute", "thunderkittens"])
    parser.add_argument("--option", default="one_shot", choices=["zero_shot", "one_shot", "few_shot"])
    parser.add_argument("--gpu-name", default="A100", help="GPU name for hardware-aware prompt, e.g. A100, H100")
    parser.add_argument("--include-hardware", action="store_true", help="Inject GPU hardware context into the prompt")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit quantization via bitsandbytes")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: {model_slug}_generated_kernels_beam next to this script)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip problems whose _v0.py already exists")
    parser.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of diverse samples to generate per problem (default: 4)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for diversity (default: 0.8)")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, f"{model_slug}_generated_kernels_beam")

    dataset_split = f"level_{args.level}"
    level_dir = f"level_{args.level}_beam"
    out_dir = os.path.join(args.output_dir, level_dir)
    os.makedirs(out_dir, exist_ok=True)

    from datasets import load_dataset

    print(f"Loading dataset {args.dataset_name} split={dataset_split} …")
    dataset = load_dataset(args.dataset_name, split=dataset_split)

    if args.all:
        problem_ids = list(range(0, len(dataset)))
    elif args.problems:
        problem_ids = parse_problems(args.problems)
    else:
        raise ValueError("Provide --problems or --all.")

    print(f"Output directory : {out_dir}")
    print(f"Problems to solve: {len(problem_ids)} problems")
    print(f"Num samples      : {args.num_samples}")

    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    model, tokenizer = load_model(args.model, load_in_4bit=args.load_in_4bit)

    for problem_id in problem_ids:
        try:
            problem = dataset[problem_id]
        except IndexError:
            print(f"[WARN] problem {problem_id} not found in level {args.level}, skipping")
            continue

        ref_arch_src = problem["code"]
        problem_name = problem.get("name", f"problem_{problem_id:04d}.py")

        if args.skip_existing and all(
            os.path.exists(
                os.path.join(out_dir, f"level_{args.level}_problem_{problem_id}_sample_{i}_kernel.py")
            )
            for i in range(args.num_samples)
        ):
            print(f"[SKIP] {problem_name}")
            continue

        print(f"\n[{problem_id}/{problem_ids[-1]}] {problem_name}")

        prompt = get_prompt_for_backend(
            ref_arch_src=ref_arch_src,
            backend=args.backend,
            option=args.option,
            include_hardware=args.include_hardware,
            gpu_name=args.gpu_name if args.include_hardware else None,
        )

        raws = generate_kernel_beam(model, tokenizer, prompt, args.max_new_tokens, args.num_samples, args.temperature)

        for beam_idx, raw in enumerate(raws):
            code = extract_code_block(raw)
            fname = f"level_{args.level}_problem_{problem_id}_sample_{beam_idx}_kernel.py"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "w") as f:
                f.write(code)
            print(f"  sample_{beam_idx} → {fname}")

        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
