"""Family 2 -- memory traffic / fusion."""

from __future__ import annotations

from conftest import src

from triton_lint.checks.family2._common import fmt_bytes, fmt_time, transfer_time

SHAPES = [((64, 64), "float32"), ((64, 64), "float32")]
NBYTES = 64 * 64 * 4

#: Big enough that memory time dominates the launch overhead (67 MB per input).
BIG_SHAPES = [((4096, 4096), "float32"), ((4096, 4096), "float32")]

TWO_ELEMENTWISE = '''
@triton.jit
def exp_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.exp(tl.load(x_ptr + offs, mask=offs < n)), mask=offs < n)

@triton.jit
def scale_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) * 2.0, mask=offs < n)
'''

REDUCE_KERNEL = '''
@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.program_id(0), tl.sum(tl.load(x_ptr + offs, mask=offs < n), axis=0))
'''


class TestF21DeadIntermediate:
    def test_fires_and_suggests_fusion_for_elementwise_chain(self, check):
        found = check(
            "F2.1",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        tmp = torch.empty_like(x)
        exp_kernel[(1,)](x, tmp, n, BLOCK=128)
        out = torch.empty_like(x)
        scale_kernel[(1,)](tmp, out, n, BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        f = found[0]
        assert f.data["fusible"] is True
        assert f.data["intermediates"] == ["tmp"]
        assert f.data["bytes"] == 2 * NBYTES  # one write + one read
        assert "Fuse" in f.message

    def test_reports_cost_but_suggests_nothing_for_reduction_to_elementwise(self, check):
        """reduction -> elementwise is only fusible if the reduced axis fits a block.
        We must NOT tell the model to fuse it (that is how KernelBenchX's refinement
        made kernels slower)."""
        found = check(
            "F2.1",
            src(
                REDUCE_KERNEL
                + TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        tmp = torch.empty_like(x)
        sum_kernel[(1,)](x, tmp, n, BLOCK=128)
        out = torch.empty_like(x)
        scale_kernel[(1,)](tmp, out, n, BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["fusible"] is False
        assert "Fuse" not in found[0].message

    def test_silent_when_intermediate_is_returned(self, fired):
        """A multi-output model must materialise it -- not dead."""
        assert not fired(
            "F2.1",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        tmp = torch.empty_like(x)
        exp_kernel[(1,)](x, tmp, n, BLOCK=128)
        out = torch.empty_like(x)
        scale_kernel[(1,)](tmp, out, n, BLOCK=128)
        return out, tmp
"""
            ),
            SHAPES,
        )

    def test_detects_intermediate_across_helper_functions(self, check):
        """The dominant real shape: one helper per kernel, intermediate crosses them."""
        found = check(
            "F2.1",
            src(
                TWO_ELEMENTWISE
                + """
def do_exp(x):
    out = torch.empty_like(x)
    exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
    return out

def do_scale(x):
    out = torch.empty_like(x)
    scale_kernel[(1,)](x, out, x.numel(), BLOCK=128)
    return out

class ModelNew(nn.Module):
    def forward(self, x):
        tmp = do_exp(x)
        return do_scale(tmp)
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["intermediates"] == ["tmp"]


class TestF22LaunchOverhead:
    def test_launch_in_loop_is_a_failure(self, check):
        found = check(
            "F2.2",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            exp_kernel[(1,)](x[i], out[i], x[i].numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        loop = [f for f in found if f.data.get("kind") == "launch_in_loop"]
        assert len(loop) == 1
        assert loop[0].severity == "fail"

    def test_single_launch_is_silent(self, fired):
        assert not fired(
            "F2.2",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )


class TestF23LayoutChurn:
    def test_fires_on_contiguous_after_permute(self, check):
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xt = x.permute(1, 0).contiguous()
        out = torch.empty_like(xt)
        exp_kernel[(1,)](xt, out, xt.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert "stride" in found[0].message

    def test_silent_on_contiguous_of_fresh_tensor(self, fired):
        """.contiguous() on an already-contiguous tensor is a no-op, not a copy."""
        assert not fired(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x).contiguous()
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )

    def test_silent_on_device_move(self, fired):
        """x.to('cuda') is not a cast and costs no pass."""
        assert not fired(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        x = x.to('cuda')
        out = torch.empty_like(x)
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )


class TestF24ZeroedOverwrittenBuffer:
    def test_fires_when_fully_overwritten(self, check):
        found = check(
            "F2.4",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.zeros_like(x)
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["bytes"] == NBYTES
        assert "empty_like" in found[0].message

    def test_silent_when_kernel_accumulates_with_atomics(self, fired):
        """zeros IS required for an atomic accumulator -- 'fixing' it would be a bug."""
        assert not fired(
            "F2.4",
            src(
                """
@triton.jit
def hist_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.atomic_add(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n))

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.zeros_like(x)
        hist_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )

    def test_silent_for_empty_like(self, fired):
        assert not fired(
            "F2.4",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )


#: Three launches -- one over the reporting threshold.
THREE_LAUNCHES = """
class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        a = torch.empty_like(x)
        exp_kernel[(1,)](x, a, n, BLOCK=128)
        b = torch.empty_like(x)
        scale_kernel[(1,)](a, b, n, BLOCK=128)
        out = torch.empty_like(x)
        exp_kernel[(1,)](b, out, n, BLOCK=128)
        return out
"""


class TestF22LaunchCount:
    def test_warns_when_launch_overhead_dominates(self, check):
        """Small problem: 3 launches cost ~15 us against ~0.02 us of memory time, so
        kernel count *is* the runtime. That regime is the actionable part."""
        found = check("F2.2", src(TWO_ELEMENTWISE + THREE_LAUNCHES), SHAPES)

        counts = [f for f in found if f.data.get("kind") == "launch_count"]
        assert len(counts) == 1
        assert counts[0].severity == "warn"
        assert counts[0].data["n_launches"] == 3
        assert counts[0].data["kernels"] == ["exp_kernel", "scale_kernel", "exp_kernel"]
        assert "launch overhead exceeds the memory-transfer time" in counts[0].message

    def test_only_informational_when_the_problem_is_memory_bound(self, check):
        """67 MB inputs: memory time swamps the launch overhead, so fusing for launch
        count alone would be the wrong advice."""
        found = check("F2.2", src(TWO_ELEMENTWISE + THREE_LAUNCHES), BIG_SHAPES)

        counts = [f for f in found if f.data.get("kind") == "launch_count"]
        assert len(counts) == 1
        assert counts[0].severity == "info"
        assert "exceeds the memory-transfer time" not in counts[0].message

    def test_no_regime_claim_without_shapes(self, check):
        """An unresolvable input shape means no byte count -- report the launches, and
        say nothing about the regime."""
        found = check(
            "F2.2",
            src(
                TWO_ELEMENTWISE
                + THREE_LAUNCHES
                + """
def get_inputs():
    return [torch.rand(n)]
"""
            ),
        )
        counts = [f for f in found if f.data.get("kind") == "launch_count"]
        assert len(counts) == 1
        assert counts[0].severity == "info"

    def test_below_threshold_is_silent(self, fired):
        assert not fired(
            "F2.2",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        tmp = torch.empty_like(x)
        exp_kernel[(1,)](x, tmp, n, BLOCK=128)
        out = torch.empty_like(x)
        scale_kernel[(1,)](tmp, out, n, BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )

    def test_launch_in_a_while_loop(self, check):
        found = check(
            "F2.2",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        i = 0
        while i < x.shape[0]:
            exp_kernel[(1,)](x[i], out[i], x[i].numel(), BLOCK=128)
            i += 1
        return out
"""
            ),
            SHAPES,
        )
        loop = [f for f in found if f.data.get("kind") == "launch_in_loop"]
        assert len(loop) == 1
        assert loop[0].severity == "fail"
        assert loop[0].data["loop_vars"] == ["while"]


class TestF21Chains:
    def test_merges_a_three_kernel_chain_into_one_suggestion(self, check):
        """Two intermediates in a row must produce "fuse these three", not two separate
        suggestions the model would act on independently."""
        found = check("F2.1", src(TWO_ELEMENTWISE + THREE_LAUNCHES), SHAPES)

        assert len(found) == 1
        assert found[0].data["intermediates"] == ["a", "b"]
        assert found[0].data["kernels"] == ["exp_kernel", "scale_kernel"]
        assert found[0].data["bytes"] == 2 * (2 * NBYTES)  # both round-trips
        assert found[0].data["fusible"] is True


class TestF23More:
    def test_fires_on_a_dtype_cast(self, check):
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xf = x.to(torch.float32)
        out = torch.empty_like(xf)
        exp_kernel[(1,)](xf, out, xf.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["op"] == "to"
        assert "tl.load(...).to(tl.float32)" in found[0].message

    def test_fires_on_a_dtype_keyword_cast(self, check):
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xf = x.to(dtype=torch.float16)
        out = torch.empty_like(xf)
        exp_kernel[(1,)](xf, out, xf.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1

    def test_silent_on_a_device_variable(self, fired):
        assert not fired(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        x = x.to(device)
        out = torch.empty_like(x)
        exp_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )

    def test_fires_on_contiguous_after_transpose_attribute(self, check):
        """x.T is a layout change, so .contiguous() on it really does copy."""
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xt = x.T.contiguous()
        out = torch.empty_like(xt)
        exp_kernel[(1,)](xt, out, xt.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["bytes"] == 2 * NBYTES

    def test_fires_on_contiguous_of_a_bound_permute(self, check):
        """The two-statement form: xt = x.permute(...); xt.contiguous().

        Regression: this was missed because the layout set is keyed by the raw scoped
        name (``forward::xt``) while the lookup canonicalised first, which resolves the
        alias back to its contiguous base (``forward::x``) and can never match.
        """
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xt = x.permute(1, 0)
        xc = xt.contiguous()
        out = torch.empty_like(xc)
        exp_kernel[(1,)](xc, out, xc.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert "stride" in found[0].message

    def test_silent_on_contiguous_of_a_slice(self, fired):
        """A slice of a contiguous tensor is contiguous -- no copy, no finding."""
        assert not fired(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xc = x[0].contiguous()
        out = torch.empty_like(xc)
        exp_kernel[(1,)](xc, out, xc.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )

    def test_fires_on_clone(self, check):
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xc = x.clone()
        out = torch.empty_like(xc)
        exp_kernel[(1,)](xc, out, xc.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["op"] == "clone"
        assert found[0].data["bytes"] == 2 * NBYTES
        assert "moves the whole tensor" in found[0].message

    def test_clone_of_an_expression_has_no_byte_count(self, check):
        """We cannot name the receiver's buffer, so we report the copy without a cost
        rather than attributing the wrong number of bytes to it."""
        found = check(
            "F2.3",
            src(
                TWO_ELEMENTWISE
                + """
class ModelNew(nn.Module):
    def forward(self, x):
        xc = (x * 2).clone()
        out = torch.empty_like(xc)
        exp_kernel[(1,)](xc, out, xc.numel(), BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert len(found) == 1
        assert found[0].data["bytes"] is None
        assert "HBM traffic" not in found[0].message


class TestFormatters:
    def test_fmt_bytes(self):
        assert fmt_bytes(512) == "512 B"
        assert fmt_bytes(2048) == "2.0 KB"
        assert fmt_bytes(3 << 20) == "3.0 MB"

    def test_fmt_time(self):
        assert fmt_time(5e-6) == "5.0 us"
        assert fmt_time(2e-3) == "2.0 ms"

    def test_transfer_time_scales_with_bytes(self):
        assert transfer_time(1_600_000_000) == 1e-3  # 1.6 GB at 1.6 TB/s
