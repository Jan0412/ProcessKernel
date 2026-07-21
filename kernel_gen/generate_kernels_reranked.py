"""
Generate kernel solutions for KernelBench problems, rerank N candidates, and keep the best.

For each problem:
  1. Generate --num-samples candidates with the LLM (held in memory).
  2. Score each candidate with a trained reranker model.
  3. Write all candidates + a scores.json to a per-problem subfolder.
  4. Write the best-scoring candidate to the flat output dir (standard naming).

Example:
    python generate_kernels_reranked.py \\
        --model Qwen/Qwen3-Coder-30B-A3B \\
        --level 1 \\
        --problems 0-49 \\
        --num-samples 8 \\
        --temperature 0.8 \\
        --reranker-checkpoint ../reranker/data/checkpoints/final \\
        --output-dir /path/to/my_run_reranked
"""

import argparse
import ast
import json
import os
import re
import sys
import textwrap

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
KERNELBENCH_SRC = os.path.join(SCRIPT_DIR, "KernelBench", "src")
sys.path.insert(0, KERNELBENCH_SRC)

from gen_config import print_generation_summary, write_generation_config

# Reranker tokenization constants — must match reranker/src/dataset.py exactly.
_RERANKER_INSTRUCTION = (
    "You are judging whether a generated GPU kernel is a correct and faster "
    "drop-in replacement for the given PyTorch reference architecture.\n"
    "Reference architecture:\n"
)
_RERANKER_SEPARATOR = "\n\nCandidate kernel:\n"


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
    """The first ```python / ```py fenced block, dedented; else the text from its first
    Python statement, with leading prose and stray fences removed.

    A newline is required after the language tag so an *indented* fence (nested under
    prose or a list item) does not have the first code line's indentation eaten while the
    rest keep theirs -- ``textwrap.dedent`` then restores a uniform block. When no fence
    is present, leading prose ("## Plan", a numbered list) is dropped by resuming at the
    first Python statement, but only when that turns an unparseable blob into a parseable
    one -- so a bare file that merely opens with a comment is returned untouched.
    """

    def _parses(src: str) -> bool:
        try:
            ast.parse(src)
            return True
        except (SyntaxError, ValueError):
            return False

    match = re.search(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text, re.DOTALL)
    if match:
        return textwrap.dedent(match.group(1)).strip()

    stripped = textwrap.dedent(re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip()))
    if not _parses(stripped):
        head = re.search(r"^[ \t]*(?:import\s|from\s|@|def\s|class\s)", stripped, re.MULTILINE)
        if head:
            candidate = textwrap.dedent(stripped[head.start():]).strip()
            if _parses(candidate):
                return candidate
    return stripped.strip()


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
    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    kwargs = dict(
        dtype="auto",
        max_model_len=16384,
        gpu_memory_utilization=0.80,
        tensor_parallel_size=num_gpus,
    )
    if load_in_4bit:
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"

    print(f"Loading model {model_id} with vLLM …")
    llm = LLM(model=model_id, **kwargs)
    print("Model loaded.")
    return llm


def generate_samples(
    llm, prompt: str, max_new_tokens: int, num_samples: int, temperature: float
) -> list[str]:
    from vllm import SamplingParams

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

    tokenizer = llm.get_tokenizer()
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=num_samples,
    )

    outputs = llm.generate([formatted_prompt], sampling_params)
    return [completion.text for completion in outputs[0].outputs]


def load_reranker(checkpoint_dir: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading reranker tokenizer from {checkpoint_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading reranker model from {checkpoint_dir} …")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.to(device)

    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Reranker loaded on {device} | VRAM used: {vram_gb:.1f} GB")
    return model, tokenizer


def score_kernels(
    reranker_model,
    reranker_tokenizer,
    ref_arch: str,
    kernel_codes: list[str],
    max_length: int,
    reserve_ref_tokens: int,
    device: str,
) -> list[float]:
    """Score each kernel against ref_arch using the reranker.

    Tokenization matches reranker/src/dataset.py:_encode_pair exactly:
    budget-based truncation that preserves up to reserve_ref_tokens of the
    reference and fills the remainder with the kernel (tail truncated).
    """
    tok = reranker_tokenizer
    eos_id = tok.eos_token_id
    instr_ids = tok.encode(_RERANKER_INSTRUCTION, add_special_tokens=False)
    sep_ids = tok.encode(_RERANKER_SEPARATOR, add_special_tokens=False)
    ref_ids = tok.encode(ref_arch, add_special_tokens=False)

    budget = max_length - (1 if eos_id is not None else 0) - len(instr_ids) - len(sep_ids)
    budget = max(budget, 0)
    ref_keep = min(len(ref_ids), reserve_ref_tokens, budget)
    ref_ids = ref_ids[:ref_keep]
    kernel_budget = max(budget - len(ref_ids), 0)

    scores = []
    for kernel in kernel_codes:
        kernel_ids = tok.encode(kernel, add_special_tokens=False)[:kernel_budget]
        ids = instr_ids + ref_ids + sep_ids + kernel_ids
        if eos_id is not None:
            ids.append(eos_id)
        input_ids = torch.tensor([ids], dtype=torch.long).to(device)
        attn_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logit = reranker_model(input_ids=input_ids, attention_mask=attn_mask).logits.squeeze(-1)
        scores.append(torch.sigmoid(logit).item())
        del input_ids, attn_mask, logit
        torch.cuda.empty_cache()

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Generate KernelBench solutions, rerank candidates, keep the best"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID, e.g. Qwen/Qwen3-Coder-30B-A3B")
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
        required=True,
        help="Output directory (a fresh empty dir; will be created if it doesn't exist)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip problems whose flat best-kernel file already exists")
    parser.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of candidates to generate per problem (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    # Reranker args
    parser.add_argument(
        "--reranker-checkpoint",
        default=None,
        help="Path to reranker checkpoint dir (relative to repo root or absolute). "
             "Default: ../reranker/data/checkpoints/final",
    )
    parser.add_argument("--reranker-max-length", type=int, default=4096, help="Max token length for reranker input")
    parser.add_argument("--reranker-device", default="cuda:0", help="Device for the reranker model (default: cuda:0)")
    parser.add_argument(
        "--reranker-reserve-ref-tokens",
        type=int,
        default=1024,
        help="Max ref-arch tokens kept in the reranker input sequence (default: 1024)",
    )
    args = parser.parse_args()

    # Resolve and load reranker first so we catch bad checkpoints before spending time loading the LLM.
    ckpt = args.reranker_checkpoint or "../reranker/data/checkpoints/final"
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(os.path.dirname(SCRIPT_DIR), ckpt)
    reranker_model, reranker_tokenizer = load_reranker(ckpt, args.reranker_device)

    dataset_split = f"level_{args.level}"
    os.makedirs(args.output_dir, exist_ok=True)

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
        run_name=os.path.basename(os.path.normpath(args.output_dir)),
        dataset_split=dataset_split,
        num_problems=len(problem_ids),
        reranker_checkpoint_resolved=ckpt,
        script=os.path.basename(__file__),
    )
    print_generation_summary(
        config,
        keys=["model", "dataset_name", "level", "num_problems", "num_samples",
              "temperature", "backend", "option", "max_new_tokens",
              "reranker_checkpoint_resolved", "reranker_max_length", "output_dir"],
        title="KernelBench generation (reranked)",
    )
    cfg_path = write_generation_config(args.output_dir, config)
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

        best_flat_path = os.path.join(
            args.output_dir,
            f"level_{args.level}_problem_{real_id}_sample_0_kernel.py",
        )
        if args.skip_existing and os.path.exists(best_flat_path):
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

        raws = generate_samples(llm, prompt, args.max_new_tokens, args.num_samples, args.temperature)
        kernel_codes = [extract_code_block(raw) for raw in raws]

        scores = score_kernels(
            reranker_model,
            reranker_tokenizer,
            ref_arch_src,
            kernel_codes,
            args.reranker_max_length,
            args.reranker_reserve_ref_tokens,
            args.reranker_device,
        )
        best_idx = scores.index(max(scores))
        print(f"  scores: {[f'{s:.3f}' for s in scores]}  → best=sample_{best_idx} ({scores[best_idx]:.3f})")

        # Write all samples + scores.json to per-problem subfolder.
        problem_dir = os.path.join(args.output_dir, f"problem_{real_id}")
        os.makedirs(problem_dir, exist_ok=True)

        scores_record = {}
        for i, code in enumerate(kernel_codes):
            fname = f"level_{args.level}_problem_{real_id}_sample_{i}_kernel.py"
            with open(os.path.join(problem_dir, fname), "w") as f:
                f.write(code)
            scores_record[fname] = scores[i]

        scores_meta = {
            "scores": scores_record,
            "best": f"level_{args.level}_problem_{real_id}_sample_{best_idx}_kernel.py",
        }
        with open(os.path.join(problem_dir, "scores.json"), "w") as f:
            json.dump(scores_meta, f, indent=2)

        # Write best-scoring sample to the flat output dir with standard naming.
        with open(best_flat_path, "w") as f:
            f.write(kernel_codes[best_idx])
        print(f"  → {os.path.basename(best_flat_path)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
