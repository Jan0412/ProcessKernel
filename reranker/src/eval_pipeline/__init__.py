"""Reranker evaluation pipeline.

Two stages, so evaluating several rerankers is cheap:

  * ``build_eval_table`` (Stage 1a, CPU) — walks the KernelBench run dirs, joins
    each kernel's eval outcome with the per-problem PyTorch baseline timings, and
    writes a reranker-independent ``eval_table.jsonl`` (one row per kernel). Built
    once and shared by every reranker.

  * ``score_run`` (Stage 1b, GPU) — reads the eval table, scores each (reference,
    kernel) pair with a trained reranker checkpoint, and writes a small
    ``scores/<name>.jsonl`` keyed by ``(run_name, kernel_file)``. Adding a new
    reranker is one GPU pass over the same eval table.

The analysis notebook (``notebooks/reranker_final_eval.ipynb``) joins any number
of scores files onto the one eval table and computes all metrics/plots.
"""
