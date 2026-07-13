"""Family 1 -- fallback / fake work.

Every check gets a true positive AND the false-positive guards that matter. The
guards are the point: a check that fires on legitimate code is worse than no check,
because the refinement loop will act on it.
"""

from __future__ import annotations

from conftest import ELEMENTWISE_KERNEL, src


class TestF11NoTritonKernel:
    def test_fires_when_no_kernel(self, fired):
        assert fired("F1.1", src("class ModelNew(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n"))

    def test_silent_when_kernel_present(self, fired):
        assert not fired("F1.1", src(ELEMENTWISE_KERNEL))


class TestF12DeadKernel:
    def test_fires_when_never_launched(self, check):
        found = check(
            "F1.2",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        return x + y
"""
            ),
        )
        assert len(found) == 1
        assert found[0].data["kernel"] == "add_kernel"

    def test_silent_when_launched_via_helper(self, fired):
        assert not fired(
            "F1.2",
            src(
                ELEMENTWISE_KERNEL
                + """
def helper(x, y):
    out = torch.empty_like(x)
    add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
    return out

class ModelNew(nn.Module):
    def forward(self, x, y):
        return helper(x, y)
"""
            ),
        )

    def test_silent_when_launched_via_submodule(self, fired):
        assert not fired(
            "F1.2",
            src(
                ELEMENTWISE_KERNEL
                + """
class AddTriton(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.add = AddTriton()
    def forward(self, x, y):
        return self.add(x, y)
"""
            ),
        )

    def test_silent_when_launched_via_autograd_function(self, fired):
        assert not fired(
            "F1.2",
            src(
                ELEMENTWISE_KERNEL
                + """
class AddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out

class ModelNew(nn.Module):
    def forward(self, x, y):
        return AddFn.apply(x, y)
"""
            ),
        )


class TestF13DiscardedOutput:
    def test_fires_when_output_thrown_away(self, check):
        found = check(
            "F1.3",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return torch.add(x, y)
"""
            ),
        )
        assert len(found) == 1
        assert found[0].data["outputs"] == ["out"]

    def test_silent_when_returned(self, fired):
        assert not fired(
            "F1.3",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out
"""
            ),
        )

    def test_silent_for_inplace_kernel(self, fired):
        """An in-place kernel writes into a forward input; that is not discarded."""
        assert not fired(
            "F1.3",
            src(
                """
@triton.jit
def inplace_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs) * 2.0, mask=offs < n)

class ModelNew(nn.Module):
    def forward(self, x):
        inplace_kernel[(1,)](x, x.numel(), BLOCK=128)
        return x
"""
            ),
        )


class TestF14TorchFallback:
    def test_fires_on_heavy_op(self, check):
        found = check(
            "F1.4",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, w):
        h = torch.conv2d(x, w)
        out = torch.empty_like(h)
        add_kernel[(1,)](h, h, out, h.numel(), BLOCK=128)
        return out
"""
            ),
        )
        assert found and found[0].severity == "fail"
        assert "torch.conv2d" in found[0].data["heavy_ops"]

    def test_silent_on_plumbing(self, fired):
        assert not fired(
            "F1.4",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        x = x.view(-1)
        y = y.reshape(x.shape)
        out = torch.empty_like(x)
        add_kernel[(triton.cdiv(x.numel(), 128),)](x, y, out, x.numel(), BLOCK=128)
        return out.view(-1)
"""
            ),
        )

    def test_silent_on_init_weight_prep(self, fired):
        """torch.* in __init__ is legitimate: weights are prepared once, not per call."""
        assert not fired(
            "F1.4",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.matmul(w, w.t()).contiguous()
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out
"""
            ),
        )

    def test_fires_on_tensor_binop(self, check):
        found = check(
            "F1.4",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out * 2.0
"""
            ),
        )
        assert any(f.data.get("kind") == "binop" for f in found)

    def test_silent_on_integer_grid_arithmetic(self, fired):
        """(n + BLOCK - 1) // BLOCK is index math on ints, not a tensor op."""
        assert not fired(
            "F1.4",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        n = x.numel()
        out = torch.empty_like(x)
        grid = ((n + 128 - 1) // 128,)
        add_kernel[grid](x, y, out, n, BLOCK=128)
        return out
"""
            ),
        )


class TestF15NnModuleCall:
    def test_fires_when_module_called(self, check):
        found = check(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.fc = nn.Linear(a, b)
    def forward(self, x):
        return self.fc(x)
"""
            ),
        )
        assert found and found[0].severity == "fail"

    def test_silent_for_weight_holder(self, fired):
        """Holding nn.Linear to own the weights is legitimate -- only calling it is not."""
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.fc = nn.Linear(a, b)
    def forward(self, x):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, self.fc.weight, out, x.numel(), BLOCK=128)
        return out
"""
            ),
        )

    def test_silent_for_dropout(self, fired):
        """nn.Dropout is an identity at eval time: it launches nothing."""
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.1)
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return self.drop(out)
"""
            ),
        )


class TestF16PassthroughKernel:
    def test_fires_on_memcpy_kernel(self, check):
        found = check(
            "F1.6",
            src(
                """
@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(out_ptr + offs, v, mask=offs < n)

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        copy_kernel[(1,)](x, out, x.numel(), BLOCK=128)
        return out
"""
            ),
        )
        assert len(found) == 1

    def test_silent_when_kernel_computes(self, fired):
        assert not fired(
            "F1.6",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out
"""
            ),
        )


class TestF17CompileOffload:
    def test_fires_on_torch_compile(self, fired):
        assert fired(
            "F1.7",
            src(
                """
class ModelNew(nn.Module):
    def forward(self, x):
        f = torch.compile(lambda t: t * 2)
        return f(x)
"""
            ),
        )

    def test_silent_otherwise(self, fired):
        assert not fired("F1.7", src(ELEMENTWISE_KERNEL))
