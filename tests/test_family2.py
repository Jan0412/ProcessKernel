"""Family 2 -- memory traffic / fusion."""

from __future__ import annotations

from conftest import src

SHAPES = [((64, 64), "float32"), ((64, 64), "float32")]
NBYTES = 64 * 64 * 4

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
