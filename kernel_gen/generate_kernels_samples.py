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
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
KERNELBENCH_SRC = os.path.join(SCRIPT_DIR, "KernelBench", "src")
sys.path.insert(0, KERNELBENCH_SRC)

from gen_config import print_generation_summary, write_generation_config


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


def problem_id_from_name(name: str, fallback: int) -> int:
    """Extract the real KernelBench problem id from the problem name.

    Names look like '19_ReLU.py' or '1_Square_matrix_multiplication_.py' — the
    leading integer is the problem id within the level, which does not match the
    dataset array index. Fall back to the array index if no prefix is found.
    """
    match = re.match(r"(\d+)", os.path.basename(name))
    return int(match.group(1)) if match else fallback


def load_model(model_id: str, load_in_4bit: bool):
    from vllm import LLM

    # The reranker is loaded first and occupies ~8 GiB. vLLM measures its budget
    # as a fraction of *total* VRAM, so utilization must stay below the free
    # fraction (≈39/47 ≈ 0.82 here) or startup fails.
    kwargs = dict(
        dtype="auto",
        max_model_len=16384,
        gpu_memory_utilization=0.80,
    )
    if load_in_4bit:
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"

    print(f"Loading model {model_id} with vLLM …")
    llm = LLM(model=model_id, **kwargs)
    print("Model loaded.")
    return llm


def generate_samples(
    llm, prompt: str, max_new_tokens: int, num_samples: int, temperature: float,
    think_temperature: float | None = None,
) -> list[str]:
    from vllm import SamplingParams

    SYSTEM_MSG = (
        "You write custom kernels to replace the pytorch operators in the given "
        "architecture to get speedups.\n\n"
        "You have complete freedom to choose the set of operators you want to replace. "
        "You may replace some operators with custom kernels and leave others unchanged.\n\n"
        "Before writing any code, first think through and lay out a plan. Identify which "
        "operators are the most promising to replace, explain why, and describe the kernel "
        "strategy you intend to use. Keep this planning section concise.\n\n"
        "After you have written out the plan, implement it. You need to provide the "
        "complete Python code wrapped in a Python code block that starts with ```python "
        "and ends with ```."
    )

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]

    tokenizer = llm.get_tokenizer()
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if think_temperature is not None:
        CODE_FENCE = "```python"
        # Instruct models ignore a "plan first" instruction and emit the code fence
        # immediately. Prefill the assistant turn with a plan heading so the model
        # is forced to start in prose, generated at think_temperature.
        PLAN_PREFIX = "## Plan\n"
        plan_prompt = formatted_prompt + PLAN_PREFIX

        # Pass 1: planning at think_temperature, stop before the code fence.
        think_params = SamplingParams(
            temperature=think_temperature,
            max_tokens=max_new_tokens,
            stop=[CODE_FENCE],
            include_stop_str_in_output=False,
            n=1,
        )
        think_outputs = llm.generate([plan_prompt] * num_samples, think_params)
        print(f"  plan lengths (chars): {[len(o.outputs[0].text) for o in think_outputs]}")

        # Pass 2: generate the code block at the lower temperature.
        output_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=1,
        )
        continuations = [
            plan_prompt + out.outputs[0].text + CODE_FENCE
            for out in think_outputs
        ]
        final_outputs = llm.generate(continuations, output_params)
        return [
            PLAN_PREFIX + out_think.outputs[0].text + CODE_FENCE + out_code.outputs[0].text
            for out_think, out_code in zip(think_outputs, final_outputs)
        ]

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=num_samples,
    )

    outputs = llm.generate([formatted_prompt], sampling_params)
    return [completion.text for completion in outputs[0].outputs]


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
    parser.add_argument("--think-temperature", type=float, default=None, help="If set, enables two-pass generation split at the ```python fence: this temperature is used for the reasoning/prose before the code, --temperature is used for the code itself")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, f"{model_slug}_generated_kernels_beam")

    dataset_split = f"level_{args.level}"
    out_dir = args.output_dir
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

    config = dict(vars(args))
    config.update(
        run_name=os.path.basename(os.path.normpath(out_dir)),
        dataset_split=dataset_split,
        num_problems=len(problem_ids),
        script=os.path.basename(__file__),
    )
    print_generation_summary(
        config,
        keys=["model", "dataset_name", "level", "num_problems", "num_samples",
              "temperature", "think_temperature", "backend", "option",
              "max_new_tokens", "output_dir"],
        title="KernelBench generation (sampling)",
    )
    cfg_path = write_generation_config(out_dir, config)
    print(f"Saved config     : {cfg_path}")

    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    llm = load_model(args.model, load_in_4bit=args.load_in_4bit)

    for problem_id in problem_ids:
        try:
            problem = dataset[problem_id]
        except IndexError:
            print(f"[WARN] problem {problem_id} not found in level {args.level}, skipping")
            continue

        ref_arch_src = problem["code"]
        problem_name = problem.get("name", f"problem_{problem_id:04d}.py")
        real_id = problem_id_from_name(problem_name, problem_id)

        if args.skip_existing and all(
            os.path.exists(
                os.path.join(out_dir, f"level_{args.level}_problem_{real_id}_sample_{i}_kernel.py")
            )
            for i in range(args.num_samples)
        ):
            print(f"[SKIP] {problem_name}")
            continue

        print(f"\n[{problem_id}/{problem_ids[-1]}] {problem_name} (problem {real_id})")

        prompt = get_prompt_for_backend(
            ref_arch_src=ref_arch_src,
            backend=args.backend,
            option=args.option,
            include_hardware=args.include_hardware,
            gpu_name=args.gpu_name if args.include_hardware else None,
        )

        raws = generate_samples(llm, prompt, args.max_new_tokens, args.num_samples, args.temperature, args.think_temperature)

        for beam_idx, raw in enumerate(raws):
            code = extract_code_block(raw)
            fname = f"level_{args.level}_problem_{real_id}_sample_{beam_idx}_kernel.py"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "w") as f:
                f.write(code)
            print(f"  sample_{beam_idx} → {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
