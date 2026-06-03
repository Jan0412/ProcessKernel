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
import gc
import json
import os
import re
import sys

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELBENCH_SRC = os.path.join(SCRIPT_DIR, "KernelBench", "src")
sys.path.insert(0, KERNELBENCH_SRC)

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
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip())


def load_model(model_id: str, load_in_4bit: bool):
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

    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded on {next(model.parameters()).device} | VRAM used: {vram_gb:.1f} GB")
    return model, tokenizer


def generate_samples(
    model, tokenizer, prompt: str, max_new_tokens: int, num_samples: int, temperature: float
) -> list[str]:
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

    print(f"Output directory : {args.output_dir}")
    print(f"Problems to solve: {len(problem_ids)} problems")
    print(f"Num samples      : {args.num_samples}")
    print(f"Reranker         : {ckpt}")

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

        best_flat_path = os.path.join(
            args.output_dir,
            f"level_{args.level}_problem_{problem_id}_sample_0_kernel.py",
        )
        if args.skip_existing and os.path.exists(best_flat_path):
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

        raws = generate_samples(model, tokenizer, prompt, args.max_new_tokens, args.num_samples, args.temperature)
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
        problem_dir = os.path.join(args.output_dir, f"problem_{problem_id}")
        os.makedirs(problem_dir, exist_ok=True)

        scores_record = {}
        for i, code in enumerate(kernel_codes):
            fname = f"level_{args.level}_problem_{problem_id}_sample_{i}_kernel.py"
            with open(os.path.join(problem_dir, fname), "w") as f:
                f.write(code)
            scores_record[fname] = scores[i]

        scores_meta = {
            "scores": scores_record,
            "best": f"level_{args.level}_problem_{problem_id}_sample_{best_idx}_kernel.py",
        }
        with open(os.path.join(problem_dir, "scores.json"), "w") as f:
            json.dump(scores_meta, f, indent=2)

        # Write best-scoring sample to the flat output dir with standard naming.
        with open(best_flat_path, "w") as f:
            f.write(kernel_codes[best_idx])
        print(f"  → {os.path.basename(best_flat_path)}")

        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
