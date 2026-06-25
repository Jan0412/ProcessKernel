"""Listwise (LambdaRank) training path for the kernel reranker.

This subpackage reuses the pointwise model, encoding, metrics, and MLflow wiring
from ``reranker.src`` and only adds what is genuinely listwise: a list-building
step (one speed-graded candidate list per problem), a list dataset/collator, and
a LambdaRank-loss trainer. The pointwise loop (``reranker.src.train``) and the
pairwise loop (``reranker.src.pairwise``) are left untouched and remain usable.
"""
