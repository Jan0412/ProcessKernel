"""Stage 1b — score an eval table with one reranker checkpoint.

Reads the reranker-independent ``eval_table.jsonl`` (Stage 1a), scores every
(reference architecture, candidate kernel) pair with a trained reranker, and
writes a small ``scores/<name>.jsonl`` — one row per kernel with the raw logit
and the sigmoid score, keyed by ``(run_name, kernel_file)`` so the notebook can
join it back onto the eval table. Adding another reranker is just another run of
this script against the same eval table.

Encoding is delegated to ``reranker/src/encoding.py::SequenceEncoder`` (the exact
format used in training) and batches are right-padded with the training collator
(``reranker/src/dataset.py::pad_sequences``), so a batched score equals the
single-example score used in ``reranker_eval_quality.ipynb``.

Example:
    python -m reranker.src.eval_pipeline.score_run \\
        --eval-table reranker/data/eval/eval_table.jsonl \\
        --runs /path/runs/gpt-oss-120b_kernelbook_level5_triton \\
        --kernelbench-dir /path/KernelBench \\
        --checkpoint /path/listwise_model_06B \\
        --name listwise_l56 \\
        --out reranker/data/eval/scores/listwise_l56.jsonl \\
        --max-length 6144 --batch-size 8
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from reranker.src.dataset import pad_sequences
from reranker.src.encoding import INSTRUCTION, SEPARATOR, SequenceEncoder
from reranker.src.data.build_dataset import expand_run_dirs
from reranker.src.eval_pipeline.common import RefArchIndex, git_sha


def load_reranker(checkpoint_dir: str, device: str):
    """Load a seq-cls reranker + tokenizer (mirrors generate_kernels_reranked.py)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading reranker tokenizer from {checkpoint_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading reranker model from {checkpoint_dir} …")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir, num_labels=1, dtype=torch.bfloat16,
    )
    # The seq-cls head pools the last non-pad token; with right-padded batches it
    # must know the pad id to locate each row's real length.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.to(device)
    print(f"Reranker loaded on {device}")
    return model, tokenizer


def _run_dir_map(
    run_dirs: list[str], rounds: list[int] | None = None
) -> dict[str, Path]:
    """``{leaf run_name: leaf dir}`` — the same expansion the eval table was built with.

    The eval table's ``run_name`` is a LEAF (``<run>__shard_00__round0``), so mapping the
    run roots by basename would match nothing and every kernel would count as missing.
    Pass the same ``--rounds`` the table was built with.
    """
    leaves = expand_run_dirs(run_dirs, [None] * len(run_dirs), rounds)
    return {name: Path(os.path.abspath(d)) for d, name, _ in leaves}


@torch.inference_mode()
def _score_batch(model, input_id_lists: list[list[int]], pad_id: int, device: str):
    """Return (logits, sigmoids) for a right-padded batch of encoded pairs."""
    input_ids, attn = pad_sequences(input_id_lists, pad_id)
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    logits = model(input_ids=input_ids, attention_mask=attn).logits.squeeze(-1).float()
    sigmoids = torch.sigmoid(logits)
    return logits.cpu().tolist(), sigmoids.cpu().tolist()


def score_run(
    eval_table: str,
    run_dirs: list[str],
    kernelbench_dir: str,
    checkpoint: str,
    name: str,
    out_path: str,
    device: str = "cuda:0",
    max_length: int = 6144,
    reserve_ref_tokens: int = 1024,
    batch_size: int = 8,
    compiled_only: bool = False,
    rounds: list[int] | None = None,
) -> str:
    rows = [json.loads(l) for l in Path(eval_table).read_text().splitlines() if l.strip()]
    if compiled_only:
        rows = [r for r in rows if r.get("compiled")]
    print(f"Scoring {len(rows)} kernels from {eval_table}"
          f"{' (compiled only)' if compiled_only else ''}")

    run_dirs_map = _run_dir_map(run_dirs, rounds)
    ref_index = RefArchIndex(kernelbench_dir)
    model, tokenizer = load_reranker(checkpoint, device)
    encoder = SequenceEncoder(tokenizer, max_length=max_length,
                              reserve_ref_tokens=reserve_ref_tokens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Cache reference token ids per (level, problem_id) — kernels of one problem
    # all share the reference, and re-encoding it per kernel is wasteful.
    n_written = 0
    n_missing_ref = 0
    n_missing_kernel = 0
    n_truncated = 0

    # Buffer of pending (row, meta, input_ids) flushed in batches; each is written
    # as soon as its batch is scored, so a crash leaves a valid partial file.
    from tqdm.auto import tqdm

    with open(out_path, "w") as out_f:
        pending_meta: list[dict] = []
        pending_ids: list[list[int]] = []

        def flush():
            nonlocal n_written
            if not pending_ids:
                return
            logits, sigmoids = _score_batch(model, pending_ids, pad_id, device)
            for rec, logit, sig in zip(pending_meta, logits, sigmoids):
                rec["score_logit"] = float(logit)
                rec["score_sigmoid"] = float(sig)
                out_f.write(json.dumps(rec) + "\n")
                n_written += 1
            out_f.flush()
            pending_meta.clear()
            pending_ids.clear()

        for r in tqdm(rows, desc="Scoring kernels", unit="kernel"):
            level, pid = int(r["level"]), int(r["problem_id"])
            ref = ref_index.source(level, pid)
            if ref is None:
                n_missing_ref += 1
                continue
            run_dir = run_dirs_map.get(r["run_name"])
            kpath = (run_dir / r["kernel_file"]) if run_dir else None
            if kpath is None or not kpath.is_file():
                n_missing_kernel += 1
                continue
            kernel_src = kpath.read_text()

            ids, meta = encoder.encode_with_meta(ref, kernel_src)
            if meta["truncated"]:
                n_truncated += 1
            pending_meta.append({
                "run_name": r["run_name"],
                "kernel_file": r["kernel_file"],
                "ref_tokens": meta["ref_tokens"],
                "kernel_tokens": meta["kernel_tokens"],
                "truncated": meta["truncated"],
            })
            pending_ids.append(ids)
            if len(pending_ids) >= batch_size:
                flush()
        flush()

    meta = {
        "name": name,
        "out": os.path.abspath(out_path),
        "eval_table": os.path.abspath(eval_table),
        "checkpoint": os.path.abspath(checkpoint),
        "runs": [os.path.abspath(d) for d in run_dirs],
        "kernelbench_dir": os.path.abspath(kernelbench_dir),
        "encoder": {
            "max_length": max_length,
            "reserve_ref_tokens": reserve_ref_tokens,
            "instruction": INSTRUCTION,
            "separator": SEPARATOR,
        },
        "compiled_only": compiled_only,
        "rounds": rounds,
        "batch_size": batch_size,
        "scored": n_written,
        "missing_ref": n_missing_ref,
        "missing_kernel_file": n_missing_kernel,
        "truncated": n_truncated,
        "git_sha": git_sha(kernelbench_dir),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print(f"scores written : {out_path}  ({n_written} rows, name={name})")
    print(f"  sidecar meta : {meta_path}")
    print(f"  truncated    : {n_truncated}/{n_written}")
    if n_missing_ref:
        print(f"  [WARN] missing ref arch   : {n_missing_ref}")
    if n_missing_kernel:
        print(f"  [WARN] missing kernel file: {n_missing_kernel}")
    print("=" * 60)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Score an eval table with a reranker (Stage 1b)")
    ap.add_argument("--eval-table", required=True, help="eval_table.jsonl from build_eval_table")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run dirs (to read kernel sources by run_name/kernel_file)")
    ap.add_argument("--kernelbench-dir", required=True,
                    help="KernelBench checkout for reference archs")
    ap.add_argument("--checkpoint", required=True, help="Reranker checkpoint dir")
    ap.add_argument("--name", required=True, help="Short reranker name (labels the scores file/meta)")
    ap.add_argument("--out", required=True, help="Output scores/<name>.jsonl path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--reserve-ref-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rounds", nargs="+", type=int, default=None,
                    help="Same --rounds the eval table was built with; omit for finals")
    ap.add_argument("--compiled-only", action="store_true",
                    help="Skip non-compiling kernels (saves GPU; §4's out-of-dist "
                         "check then has no non-compiled rows)")
    args = ap.parse_args()
    score_run(
        eval_table=args.eval_table,
        run_dirs=args.runs,
        kernelbench_dir=args.kernelbench_dir,
        checkpoint=args.checkpoint,
        name=args.name,
        out_path=args.out,
        device=args.device,
        max_length=args.max_length,
        reserve_ref_tokens=args.reserve_ref_tokens,
        batch_size=args.batch_size,
        compiled_only=args.compiled_only,
        rounds=args.rounds,
    )


if __name__ == "__main__":
    main()
