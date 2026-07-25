"""Metamorphic relations for ``derive_scalars`` (gemtest).

Entropy, DeepConf confidence and tail mass are **symmetric functions of the top-K row**:
they depend on the multiset of probabilities, not on the order the alternatives arrived
in. vLLM's row order is already known to be surprising (sampled-token-first, not
best-first), so a scalar that secretly depended on column order would be wrong on real
data in a way no fixed example might reveal. The relation: **permuting the columns of
every row must leave the order-independent scalars unchanged.**

(``top1_lp`` and ``margin`` are deliberately excluded -- they DO depend on order, and
that is correct. That asymmetry is itself the reason this MR is worth stating.)
"""

from __future__ import annotations

import gemtest as gmt
import numpy as np

from kernel_gen.core.trace import derive_scalars

_ORDER_INDEPENDENT = ("entropy", "deepconf_c", "tail_mass")
_rng = np.random.default_rng(0)


def _logprob_matrix(n_tokens: int, k: int) -> np.ndarray:
    """A [T, K] top-K logprob matrix shaped like real output: descending per row."""
    lp = -np.abs(_rng.gamma(2.0, 1.5, size=(n_tokens, k)))
    return np.sort(lp, axis=1)[:, ::-1].astype(np.float32)


MATRICES = [_logprob_matrix(t, k) for t, k in [(8, 20), (5, 10), (12, 20), (3, 5), (20, 20)]]


permute_mr = gmt.create_metamorphic_relation(name="topk_column_permutation", data=MATRICES)


@gmt.transformation(permute_mr)
def shuffle_each_rows_columns(topk_lp: np.ndarray) -> np.ndarray:
    out = topk_lp.copy()
    for row in out:
        _rng.shuffle(row)  # in place, per row -- a different permutation each row
    return out


@gmt.relation(permute_mr)
def order_independent_scalars_unchanged(source_out: dict, followup_out: dict) -> bool:
    return all(
        np.allclose(source_out[name], followup_out[name], atol=1e-4, rtol=1e-4)
        for name in _ORDER_INDEPENDENT
    )


@gmt.system_under_test(permute_mr)
def test_symmetric_scalars_ignore_topk_order(topk_lp: np.ndarray) -> dict:
    return derive_scalars(topk_lp)
