"""Family 1 ground truth from the 2026-07-13 hand audit."""


from real_samples._loader import findings, full_report

# ---------------------------------------------------------------------------
# Known-clean anchor: pure-Triton Mish with a torch backward (p32_s6). This is
# the sample whose false F1.4 flag started the audit.
# ---------------------------------------------------------------------------


def test_clean_mish_sample_has_no_findings():
    report = full_report(32, 6)
    assert report.findings == []


# ---------------------------------------------------------------------------
# F1.1 no_triton_kernel
# ---------------------------------------------------------------------------


def test_f1_1_truncated_generation():
    # 7-line fragment referencing a wrapper that was never generated.
    assert [f.severity for f in findings(2595, 8, "F1.1")] == ["fail"]


def test_f1_1_comment_only_file():
    # The whole generation is the string "# code".
    assert [f.severity for f in findings(16114, 4, "F1.1")] == ["fail"]


def test_f1_1_nested_kernels_are_seen():
    assert findings(349, 1, "F1.1") == []


def test_f1_4_nested_kernel_body_not_scanned_as_host():
    tl_ops = [
        op
        for f in findings(349, 1, "F1.4")
        for op in f.data.get("ops", [])
        if op.startswith("tl.")
    ]
    assert tl_ops == []


def test_f1_5_on_nested_kernel_file_is_real():
    # Independent of BUG-1: self.mean / self.log_std are nn.Linear modules
    # invoked in forward. Verified real.
    assert any(f.severity == "fail" for f in findings(349, 1, "F1.5"))


# ---------------------------------------------------------------------------
# F1.2 dead_kernel
# ---------------------------------------------------------------------------


def test_f1_2_loss_kernel_never_called_from_forward():
    # cross_time_loss_kernel is reachable only via cross_time_steps_loss(),
    # which forward() never calls. Verified dead.
    fs = findings(7230, 7, "F1.2")
    assert [f.severity for f in fs] == ["fail"]
    assert "cross_time_loss_kernel" in fs[0].message


def test_f1_2_predict_only_kernels_are_dead():
    # triton_sigmoid / triton_ge are called only from predict(), not forward().
    fs = findings(17660, 7, "F1.2")
    assert sorted(f.severity for f in fs) == ["fail", "fail"]
    text = " ".join(f.message for f in fs)
    assert "sigmoid_kernel" in text and "ge_kernel" in text


def test_f1_2_stub_dispatch_kernels_are_dead():
    # forward() reaches a staticmethod stub that raises NotImplementedError;
    # the Triton path hangs off the wrong class and never runs.
    fs = findings(15947, 7, "F1.2")
    assert sorted(f.severity for f in fs) == ["fail", "fail"]


def test_f1_2_truncated_file_downgrades_to_info():
    # 8-line fragment: a kernel def and nothing else -- no entry point, so
    # unreachability cannot be proven.
    assert [f.severity for f in findings(4053, 0, "F1.2")] == ["info"]


def test_f1_2_default_arg_activation_is_live():
    assert findings(6227, 1, "F1.2") == []


def test_f1_4_on_default_arg_file_is_real():
    # Independent of BUG-2: the attention block computes torch.matmul and
    # F.softmax in forward. Verified real.
    fs = findings(6227, 1, "F1.4")
    heavy = {op for f in fs for op in f.data.get("heavy_ops", [])}
    assert "torch.matmul" in heavy and "F.softmax" in heavy


# ---------------------------------------------------------------------------
# F1.3 discarded_output -- the audit found 4/4 sampled findings false.
# ---------------------------------------------------------------------------


def test_f1_3_se_block_pipeline_outputs_are_consumed():
    assert findings(2983, 0, "F1.3") == []


def test_f1_3_scalar_accumulator_is_read_by_host():
    assert findings(3840, 7, "F1.3") == []


def test_f1_3_bce_accumulator_is_read_by_host():
    assert findings(4984, 6, "F1.3") == []


def test_f1_3_sliced_output_is_used():
    assert findings(15162, 3, "F1.3") == []


# ---------------------------------------------------------------------------
# F1.4 torch_fallback -- verified real on these samples.
# ---------------------------------------------------------------------------


def test_f1_4_attention_softmax_fallback():
    fs = findings(621, 9, "F1.4")
    heavy = {op for f in fs for op in f.data.get("heavy_ops", [])}
    assert "self.softmax" in heavy
    assert any(f.severity == "fail" for f in fs)


def test_f1_4_masked_attention_softmax_fallback():
    fs = findings(659, 6, "F1.4")
    heavy = {op for f in fs for op in f.data.get("heavy_ops", [])}
    assert "self.softmax" in heavy


def test_f1_4_matmul_operator_fallback():
    # Adjacency/permutation transforms computed with the @ operator in forward.
    fs = [f for f in findings(1970, 0, "F1.4") if f.data.get("kind") == "binop"]
    assert len(fs) == 1
    assert "@" in fs[0].data["ops"]
    assert fs[0].severity == "fail"


def test_f1_4_light_gelu_is_warn_not_fail():
    fs = findings(5100, 1, "F1.4")
    assert {f.severity for f in fs} == {"warn"}
    ops = {op for f in fs for op in f.data["ops"]}
    assert "F.gelu" in ops


def test_f1_4_nalu_light_fallbacks_are_real():
    # torch.sigmoid / .log / .abs / torch.exp around the Triton matmuls.
    fs = findings(7092, 8, "F1.4")
    ops = {op for f in fs for op in f.data["ops"]}
    assert "torch.sigmoid" in ops and "torch.exp" in ops


def test_f1_4_numel_division_is_not_tensor_arithmetic():
    ops = {op for f in findings(7092, 8, "F1.4") for op in f.data["ops"]}
    assert "//" not in ops


# ---------------------------------------------------------------------------
# F1.5 nn_module_call -- verified real on all four samples.
# ---------------------------------------------------------------------------


def test_f1_5_conv_module_invoked():
    assert any(f.severity == "fail" for f in findings(777, 3, "F1.5"))


def test_f1_5_linear_gates_invoked():
    fs = findings(13039, 3, "F1.5")
    assert [f.severity for f in fs] == ["fail"]
    attrs = {m["attr"] for f in fs for m in f.data["modules"]}
    assert {"z_gate", "r_gate", "f_gate"} <= attrs


def test_f1_5_light_leakyrelu_is_warn():
    assert [f.severity for f in findings(13889, 2, "F1.5")] == ["warn"]


def test_f1_5_convtranspose_stack_invoked():
    fs = findings(15662, 8, "F1.5")
    assert [f.severity for f in fs] == ["fail"]
    assert "ConvTranspose2d" in " ".join(fs[0].data["heavy"])


# ---------------------------------------------------------------------------
# F1.6 passthrough_kernel -- the audit found 3/3 sampled findings false.
# ---------------------------------------------------------------------------


def test_f1_6_constexpr_dispatch_kernel_computes():
    assert findings(4141, 3, "F1.6") == []


def test_f1_6_patch_merge_kernel_is_not_a_decoy():
    assert findings(14369, 0, "F1.6") == []


def test_f1_6_cat_kernel_is_not_a_decoy():
    assert findings(15689, 5, "F1.6") == []


# ---------------------------------------------------------------------------
# New bugs found in a fresh audit of runs/Qwen3-Coder-Next_kernelbook_level5_triton
# (BUG-14, BUG-15, BUG-16). See BUGS.md.
# ---------------------------------------------------------------------------


def test_f1_2_triton_device_function_is_not_dead():
    assert findings(11155, 7, "F1.2") == []


def test_f1_3_partial_sums_reduced_by_torch_sum_is_a_use():
    # The multi-block branch does `total_sum = torch.sum(partial_sums)` -- a genuine
    # host read. We assert only on `partial_sums` (not `== []`) because the sibling
    # `result` finding is a separate BUG-4 subscript read; isolating `partial_sums`
    # keeps this test tied to BUG-15 alone.
    discarded = {out for f in findings(12206, 9, "F1.3") for out in f.data.get("outputs", [])}
    assert "partial_sums" not in discarded


def test_f1_4_local_triton_submodule_is_not_a_fallback():
    assert findings(10000, 3, "F1.4") == []
