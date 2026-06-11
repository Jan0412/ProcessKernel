"""Pairwise (ranking) training path for the kernel reranker.

This subpackage reuses the pointwise model, encoding, metrics, and MLflow wiring
from ``reranker.src`` and only adds what is genuinely pairwise: a pair-building
step, a pair dataset/collator, and a pairwise-loss trainer. The pointwise loop
(``reranker.src.train``) is left untouched and remains fully usable.
"""
