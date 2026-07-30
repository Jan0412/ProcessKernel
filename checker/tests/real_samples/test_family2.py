"""Family 2 ground truth from the 2026-07-13 hand audit."""


import pytest

from real_samples._loader import findings

# ---------------------------------------------------------------------------
# F2.1 dead_intermediate
# ---------------------------------------------------------------------------


def test_f2_1_clamp_scale_intermediate():
    # `scale`: reduction -> elementwise, never escapes. Verified real; the
    # non-fusible pair correctly stays at info with no fuse instruction.
    fs = findings(10292, 4, "F2.1")
    assert [f.severity for f in fs] == ["info"]
    assert "scale" in fs[0].data["intermediates"]


def test_f2_1_cov2corr_intermediate():
    # `stds` is written by diag_sqrt_kernel and read only by corr_kernel.
    fs = findings(16407, 2, "F2.1")
    assert [f.severity for f in fs] == ["warn"]
    assert "stds" in fs[0].data["intermediates"]


def test_f2_1_distance_pipeline_norms_are_real():
    # xnorm/ynorm feed only the distance kernel. Verified real.
    names = {n for f in findings(5270, 3, "F2.1") for n in f.data["intermediates"]}
    assert "xnorm" in names


def test_f2_1_bandwidth_input_is_not_dead():
    names = {n for f in findings(5270, 3, "F2.1") for n in f.data["intermediates"]}
    assert "dnorm2" not in names


def test_f2_1_unet_skip_connections_are_not_dead():
    names = {n for f in findings(12761, 4, "F2.1") for n in f.data["intermediates"]}
    assert "enc1" not in names


# ---------------------------------------------------------------------------
# F2.2 launch_in_loop / launch_count -- verified real on all four samples.
# ---------------------------------------------------------------------------


def test_f2_2_batched_matmul_loop():
    fs = [f for f in findings(9332, 4, "F2.2") if f.data.get("kind") == "launch_in_loop"]
    assert [f.severity for f in fs] == ["fail"]
    assert fs[0].data["kernel"] == "matmul_kernel"


def test_f2_2_gram_matrix_loop():
    fs = [f for f in findings(15208, 1, "F2.2") if f.data.get("kind") == "launch_in_loop"]
    assert [f.severity for f in fs] == ["fail"]
    assert fs[0].data["kernel"] == "gram_kernel"


def test_f2_2_three_launch_count_reported():
    fs = [f for f in findings(241, 1, "F2.2") if f.data.get("kind") == "launch_count"]
    assert len(fs) == 1
    assert fs[0].data["n_launches"] == 3


def test_f2_2_launch_bound_regime():
    fs = [f for f in findings(17913, 2, "F2.2") if f.data.get("kind") == "launch_count"]
    assert len(fs) == 1
    assert fs[0].data["n_launches"] == 3
    # warn when shapes resolve (launch overhead > memory time), info otherwise;
    # shape fallback depends on the KernelBench reference being present.
    assert fs[0].severity in ("warn", "info")


# ---------------------------------------------------------------------------
# F2.3 layout_churn -- verified real on all four samples.
# ---------------------------------------------------------------------------


def test_f2_3_chained_transpose_contiguous():
    # weight.t().contiguous() -- a provably non-contiguous receiver. The plain
    # weight.contiguous() four lines above is correctly NOT flagged.
    fs = findings(3502, 3, "F2.3")
    assert [f.data["op"] for f in fs] == ["contiguous"]


def test_f2_3_dtype_cast():
    fs = findings(14211, 7, "F2.3")
    assert [f.data["op"] for f in fs] == ["to"]


def test_f2_3_permute_contiguous_sandwich():
    # (N, C, W) -> permute -> contiguous before and after the LayerNorm kernel.
    fs = findings(15514, 3, "F2.3")
    assert [f.data["op"] for f in fs] == ["contiguous", "contiguous"]


def test_f2_3_torch_cat_copies():
    fs = findings(16248, 5, "F2.3")
    assert [f.data["op"] for f in fs] == ["cat", "cat"]


# ---------------------------------------------------------------------------
# F2.4 zeroed_overwritten_buffer
# ---------------------------------------------------------------------------


def test_f2_4_full_loss_memset_is_not_wasted():
    # Re-audited: the original ground truth (a wasted memset) was wrong. `full_loss`
    # is `torch.zeros(N)` filled by a *host* scatter `full_loss[mask] = loss` (only the
    # non-ignored positions), and its zeros are what represent the ignored-index entries
    # -- `empty_like` would leave those as garbage. It is also read on the host
    # (`loss = full_loss; loss = loss * ratio`), so the kernel does not own it. F2.4 must
    # stay silent; firing here was a false positive with correctness-breaking advice.
    assert findings(9275, 5, "F2.4") == []


def test_f2_4_diagonal_store_keeps_zero_init():
    assert findings(7090, 0, "F2.4") == []


def test_f2_4_sparse_branch_accumulator_keeps_zero_init():
    assert findings(13416, 9, "F2.4") == []


def test_f2_4_gather_conv_transpose_zeros_are_really_wasted():
    # True positive, and a boundary control for the BUG-32/BUG-36 fixes: this
    # conv_transpose grids over EVERY output position and gathers its contributors
    # (`tl.store(out_base + y_out * out_stride_h + x_out, acc)`, no mask, full coverage),
    # so the `torch.zeros` really is a wasted memset. A fix that stops flagging masked /
    # subset writes must not silence this genuine full-overwrite case.
    fs = findings(75, 0, "F2.4", level=1)
    assert [(f.severity, f.data["buffer"]) for f in fs] == [("warn", "out")]


# ---------------------------------------------------------------------------
# BUG-32 -- open. See tests/BUGS.md. F2.4 assumes a store with no atomic overwrites
# every element, but a scatter / pad / one-hot kernel writes only a subset of the
# output and needs the zeros for the rest; `empty_like` would corrupt it.
# ---------------------------------------------------------------------------


def test_f2_4_one_hot_scatter_needs_its_zero_init():
    assert findings(1229, 4, "F2.4") == []


def test_f2_4_pad_into_larger_output_needs_its_zero_init():
    assert findings(15104, 6, "F2.4") == []


def test_f2_4_avg_pool_pad_needs_its_zero_init():
    assert findings(17549, 4, "F2.4") == []


# ---------------------------------------------------------------------------
# BUG-36 -- open. See tests/BUGS.md. A triangular matmul addresses the whole (N, N)
# output but masks the store to one triangle; the other triangle must stay at its
# `torch.zeros` value. Unlike BUG-32 the store's index space is NOT a subset, so
# `_is_partial_coverage` sees full coverage and F2.4 fires.
# ---------------------------------------------------------------------------


def test_f2_4_triangular_matmul_needs_its_zero_init():
    assert findings(15, 1, "F2.4", level=1) == []


def test_f2_4_triu_pure_mask_matmul_needs_its_zero_init():
    assert findings(14, 9, "F2.4", level=1) == []


def test_f2_4_upper_tri_strided_matmul_needs_its_zero_init():
    assert findings(14, 7, "F2.4", level=1) == []


def test_f2_4_lower_tri_zeros_like_matmul_needs_its_zero_init():
    assert findings(15, 5, "F2.4", level=1) == []


# ---------------------------------------------------------------------------
# BUG-33 -- open. See tests/BUGS.md. F2.2's recurrence guard only recognises the
# host-rebind spelling, so an in-place carried dependency is told to parallelise.
# ---------------------------------------------------------------------------


def test_f2_2_inplace_sinkhorn_iteration_is_not_told_to_grid_the_loop():
    fs = [
        f
        for f in findings(6918, 3, "F2.2")
        if f.data.get("kind") == "launch_in_loop"
        and f.data.get("kernel") == "sinkhorn_iter_kernel"
    ]
    assert fs and "into the launch grid" not in fs[0].message


def test_f2_1_maxpool_chain_is_not_reported_fusible():
    fs = findings(3952, 9, "F2.1")
    assert all(f.data["fusible"] is False for f in fs)


def test_f2_1_diamond_is_a_single_finding():
    fs = findings(2084, 2, "F2.1")
    kernels = [k for f in fs for k in f.data["kernels"]]
    # bmm_kernel is launched at one site; it must not appear in two findings.
    assert kernels.count("bmm_kernel") <= 1
