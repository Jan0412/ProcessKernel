"""Family 1 -- fallback / fake work.

Every check gets a true positive AND the false-positive guards that matter. The
guards are the point: a check that fires on legitimate code is worse than no check,
because the refinement loop will act on it.
"""

from __future__ import annotations

from conftest import ELEMENTWISE_KERNEL, src

from triton_lint.checks.family1 import f1_1_no_triton_kernel, f1_4_torch_fallback
from triton_lint.model import ModuleModel


class TestF11NoTritonKernel:
    def test_fires_when_no_kernel(self, fired):
        assert fired("F1.1", src("class ModelNew(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n"))

    def test_silent_when_kernel_present(self, fired):
        assert not fired("F1.1", src(ELEMENTWISE_KERNEL))

    def test_silent_on_an_unparseable_file(self):
        """A file we could not parse has no evidence either way -- never accuse it."""
        model = ModuleModel(parse_status="syntax_error")
        assert f1_1_no_triton_kernel.check(model) == []


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


class TestF12Guards:
    def test_silent_without_any_kernel(self, fired):
        """F1.1 already says "no kernel"; F1.2 must not pile on."""
        assert not fired(
            "F1.2",
            src("class ModelNew(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n"),
        )


class TestF13Guards:
    def test_silent_when_the_kernel_writes_nothing(self, fired):
        """No store target -- there is no output to discard. Not this check's business."""
        assert not fired(
            "F1.3",
            src(
                """
@triton.jit
def probe_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)

class ModelNew(nn.Module):
    def forward(self, x):
        probe_kernel[(1,)](x, x.numel(), BLOCK=128)
        return x
"""
            ),
        )

    def test_silent_when_the_output_argument_is_unresolvable(self, fired):
        """The launch omits out_ptr entirely. We resolved no output buffer, so we stay
        quiet rather than invent a finding."""
        assert not fired(
            "F1.3",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        add_kernel[(1,)](x, y)
        return torch.relu(x)
"""
            ),
        )


class TestF14Guards:
    def test_light_ops_only_are_a_warning(self, check):
        """No heavy op in sight: worth folding in, but not the "you didn't do the task"
        failure that a matmul fallback is."""
        found = check(
            "F1.4",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return torch.relu(out)
"""
            ),
        )
        assert len(found) == 1
        assert found[0].severity == "warn"
        assert found[0].data["heavy_ops"] == []
        assert "still uses PyTorch operators" in found[0].message

    def test_ignores_operators_that_are_not_tensor_arithmetic(self, fired):
        """`n << 1` is grid arithmetic on an int, not a PyTorch kernel."""
        assert not fired(
            "F1.4",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        n = x.numel()
        blocks = n << 1
        out = torch.empty_like(x)
        add_kernel[(blocks,)](x, y, out, n, BLOCK=128)
        return out
"""
            ),
        )

    def test_binop_with_a_scalar_operand(self, check):
        """2.0 * x -- the left operand is a constant, the right is a tensor."""
        found = check(
            "F1.4",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return 2.0 * out
"""
            ),
        )
        binops = [f for f in found if f.data.get("kind") == "binop"]
        assert len(binops) == 1
        assert binops[0].data["ops"] == ["*"]

    def test_silent_on_an_unparsed_module(self):
        assert f1_4_torch_fallback.check(ModuleModel(parse_status="syntax_error")) == []


class TestF15Guards:
    def test_light_module_is_a_warning_not_a_failure(self, check):
        """nn.ReLU does compute something, but it is not where the task's cost lives."""
        found = check(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return self.act(out)
"""
            ),
        )
        assert len(found) == 1
        assert found[0].severity == "warn"
        assert found[0].data["heavy"] == []
        assert "still applies PyTorch modules" in found[0].message

    def test_ignores_calls_to_the_models_own_methods(self, fired):
        """self._run(...) is not an nn module -- only self.<module>(...) counts."""
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.1)

    def _run(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out

    def forward(self, x, y):
        return self._run(x, y)
"""
            ),
        )
