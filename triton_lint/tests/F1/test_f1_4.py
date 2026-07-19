"""F1.4 torch_fallback.

Severity contract: the op lists decide heavy-vs-light; the functional spelling
(`F.foo`) decides compute-vs-plumbing. `fail` ("the dominant cost") is reserved
for HEAVY_OPS. Only code the timed forward() executes is scanned.
"""

from __future__ import annotations

import pytest

from conftest import ELEMENTWISE_KERNEL, src
from helpers import forward_with, lint, lint_raw

from triton_lint import build_model
from triton_lint.checks.family1 import f1_4_torch_fallback
from triton_lint.model import ModuleModel


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


# ---------------------------------------------------------------------------
# Timed scopes: autograd backward/jvp/vmap never run under the timed forward.
# ---------------------------------------------------------------------------

AUTOGRAD = '''
class MishFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sp = torch.nn.functional.softplus(x)
        return grad_output * torch.tanh(sp) * torch.sigmoid(x)


class ModelNew(nn.Module):
    def forward(self, x):
        return MishFn.apply(x)
'''


def test_torch_in_autograd_backward_is_not_a_fallback():
    assert lint(AUTOGRAD, "F1.4") == []


SUBMODULE_HELPER = '''
class Sub(nn.Module):
    def forward(self, x):
        return self._helper(x)

    def _helper(self, x):
        return torch.conv2d(x, x)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = Sub()

    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return self.sub(out)
'''


def test_self_helper_resolves_against_enclosing_class():
    # `self._helper()` inside Sub.forward must resolve to Sub._helper, not
    # ModelNew._helper -- the precise walk has no all-methods expansion to
    # paper over a bad resolution.
    findings = lint(SUBMODULE_HELPER, "F1.4")
    assert [f.severity for f in findings] == ["fail"]
    assert findings[0].data["heavy_ops"] == ["torch.conv2d"]


# ---------------------------------------------------------------------------
# Severity grading across spellings.
# ---------------------------------------------------------------------------


def test_functional_light_op_downgrades_to_warn():
    findings = lint(forward_with("F.softplus(out)"), "F1.4")
    assert [f.severity for f in findings] == ["warn"]
    assert findings[0].data["heavy_ops"] == []


def test_light_list_beats_functional_spelling():
    findings = lint(forward_with("F.relu(out)"), "F1.4")
    assert [f.severity for f in findings] == ["warn"]


def test_torch_spelling_of_activation_no_longer_silent():
    findings = lint(forward_with("torch.softplus(out)"), "F1.4")
    assert [f.severity for f in findings] == ["warn"]


def test_unknown_functional_op_is_warn_and_tagged():
    findings = lint(forward_with("F.frobnicate(out)"), "F1.4")
    assert [f.severity for f in findings] == ["warn"]
    assert findings[0].data["unknown_ops"] == ["F.frobnicate"]


def test_private_functional_namespace_ignored():
    body = '''
class ModelNew(nn.Module):
    def forward(self, x):
        reduction = F._Reduction.get_enum("mean")
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''
    assert lint(body, "F1.4") == []


def test_adaptive_pool_is_heavy():
    findings = lint(forward_with("F.adaptive_avg_pool2d(out, 1)"), "F1.4")
    assert [f.severity for f in findings] == ["fail"]


def test_functional_unfold_is_heavy_but_tensor_method_is_a_view():
    findings = lint(forward_with("F.unfold(out, 3)"), "F1.4")
    assert [f.severity for f in findings] == ["fail"]
    assert lint(forward_with("out.unfold(0, 2, 1)"), "F1.4") == []


# ---------------------------------------------------------------------------
# Regression tests for former linter bugs, now fixed (history in tests/BUGS.md).
# ---------------------------------------------------------------------------

NUMEL_ARITHMETIC = '''
class ModelNew(nn.Module):
    def forward(self, x):
        n_rows = x.numel() // 4
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out
'''


def test_numel_arithmetic_is_not_tensor_arithmetic():
    assert lint(NUMEL_ARITHMETIC, "F1.4") == []


NESTED_KERNEL_DOT = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl


def launch_matmul(a, b):
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device)

    @triton.jit
    def mm_kernel(a_ptr, b_ptr, o_ptr, M, N, K, BLOCK: tl.constexpr):
        acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        a_tile = tl.load(a_ptr + tl.arange(0, BLOCK)[:, None] * K + tl.arange(0, BLOCK)[None, :])
        b_tile = tl.load(b_ptr + tl.arange(0, BLOCK)[:, None] * N + tl.arange(0, BLOCK)[None, :])
        acc += tl.dot(a_tile, b_tile)
        tl.store(o_ptr + tl.arange(0, BLOCK)[:, None] * N + tl.arange(0, BLOCK)[None, :], acc)

    mm_kernel[(1,)](a, b, out, a.shape[0], b.shape[1], a.shape[1], BLOCK=32)
    return out


class ModelNew(nn.Module):
    def forward(self, a, b):
        return launch_matmul(a, b)
'''


def test_nested_kernel_body_is_not_host_code():
    findings = lint_raw(NESTED_KERNEL_DOT, "F1.4")
    tl_ops = [op for f in findings for op in f.data.get("ops", []) if op.startswith("tl.")]
    assert tl_ops == []


# F1.4 reduces every call to its last name segment and matches it against the op
# lists, without checking the call actually targets torch. So a call to code the
# model wrote itself -- a `self.<submodule>()` invoking a local Triton module, or a
# module-level helper function -- is reported as a PyTorch fallback whenever the
# attribute/function name happens to collide with an op token (`layer_norm`,
# `softmax`, `linear`, ...). The Triton work the model did correctly is graded a
# `fail` and the advice tells it to rewrite what is already a kernel.
LOCAL_SUBMODULE = '''
class LayerNormTriton(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = LayerNormTriton()

    def forward(self, x):
        return self.layer_norm(x)
'''

LOCAL_HELPER_FN = '''
def softmax(x):
    out = torch.empty_like(x)
    work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
    return out


class ModelNew(nn.Module):
    def forward(self, x):
        return softmax(x)
'''


def test_real_torch_op_still_fails():
    # Control for BUG-16: an actual `torch.layer_norm(...)` in forward is a genuine
    # heavy fallback and must keep failing. The fix has to stay name-blind only for
    # calls the model defined itself, not for real torch ops.
    body = '''
class ModelNew(nn.Module):
    def forward(self, x, w, b):
        h = torch.layer_norm(x, x.shape[-1:], w, b)
        out = torch.empty_like(h)
        work_kernel[(1,)](h, out, h.numel(), BLOCK=1024)
        return out
'''
    findings = lint(body, "F1.4")
    assert [f.severity for f in findings] == ["fail"]
    assert findings[0].data["heavy_ops"] == ["torch.layer_norm"]


def test_tensor_method_op_still_fires():
    # Control for BUG-16: `x.softmax(dim=1)` on a forward-input tensor really does
    # run torch's softmax, so it must keep firing. A fix that silenced all method
    # calls to dodge the self.<submodule> false positive would break this.
    body = '''
class ModelNew(nn.Module):
    def forward(self, x):
        h = x.softmax(dim=1)
        out = torch.empty_like(h)
        work_kernel[(1,)](h, out, h.numel(), BLOCK=1024)
        return out
'''
    findings = lint(body, "F1.4")
    assert [f.severity for f in findings] == ["fail"]
    assert findings[0].data["heavy_ops"] == ["x.softmax"]


def test_local_triton_submodule_is_not_a_fallback():
    assert lint(LOCAL_SUBMODULE, "F1.4") == []


def test_local_helper_function_is_not_a_fallback():
    assert lint(LOCAL_HELPER_FN, "F1.4") == []


# BUG-24: `_is_local_call` consults `model.nn_modules_in_init` *before* `attr_classes`
# and returns False on a hit, so a poisoned entry overrides the local-submodule guard
# above and re-opens BUG-16. The table is keyed on the bare attribute name for the
# whole file and is filled from every class's __init__ regardless of reachability, so
# a dead reference class -- which these generations routinely keep beside ModelNew --
# is enough to poison it.
DEAD_REFERENCE_CLASS = '''
class Reference(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n)

    def forward(self, x):
        return self.layer_norm(x)

'''


@pytest.mark.xfail(
    strict=True,
    reason="BUG-24: appending an unreachable reference class that binds "
    "`self.layer_norm = nn.LayerNorm(n)` poisons the module-wide nn_modules_in_init, "
    "so _is_local_call returns False for ModelNew's own LayerNormTriton and F1.4 "
    "grades it a heavy fallback at fail -- re-opening BUG-16 for this spelling. The "
    "class never runs; test_local_triton_submodule_is_not_a_fallback is the same file "
    "without it and passes",
)
def test_dead_reference_class_does_not_make_a_local_submodule_a_fallback():
    assert lint(LOCAL_SUBMODULE + DEAD_REFERENCE_CLASS, "F1.4") == []


def test_control_attr_classes_still_resolves_the_local_submodule():
    """The linter *knows* the attr is a local class -- F1.4 just never looks."""
    model = build_model(src(ELEMENTWISE_KERNEL + LOCAL_SUBMODULE + DEAD_REFERENCE_CLASS), "<t>")
    assert model.attr_classes["layer_norm"] == ["LayerNormTriton"]


# BUG-26: BUG-16 diagnosed `_op_of` as matching op tokens "without checking the call
# actually targets torch", and the fix (`_is_local_call`) whitelisted the code the
# model wrote itself. A `tl.*` call is neither torch nor model-authored, so it still
# falls through to the op-list match and is graded a PyTorch fallback.
TL_IN_HOST = '''
class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return REPL
'''


@pytest.mark.xfail(
    strict=True,
    reason="BUG-26: `tl.sum` is a Triton builtin, not a PyTorch operator. Telling the "
    "model to 'fold it into the Triton kernel so no work is left to PyTorch' describes "
    "an op already spelled `tl.` and is not actionable -- p98_s6 of the Qwen3-Coder "
    "lintloop run carried it through every round of the loop unchanged",
)
def test_triton_builtin_in_host_code_is_not_a_torch_fallback():
    assert lint(TL_IN_HOST.replace("REPL", "tl.sum(out)"), "F1.4") == []


@pytest.mark.xfail(
    strict=True,
    reason="BUG-26 severity face: `tl.dot` reduces to the op token `dot`, which is in "
    "HEAVY_OPS, so a Triton builtin is reported at fail as 'the dominant cost of the "
    "task -- it must be implemented as a Triton kernel'. Real sample p15299_s5 forgot "
    "the @triton.jit decorator, so F1.1 correctly fails it; this co-finding points at "
    "the tl.dot inside the kernel body and can send the model rewriting the kernel "
    "rather than adding the decorator",
)
def test_triton_dot_in_host_code_is_not_a_heavy_torch_fallback():
    assert lint(TL_IN_HOST.replace("REPL", "tl.dot(out, out)"), "F1.4") == []


def test_control_torch_sum_in_the_same_position_still_fires():
    """Pins BUG-26 to the `tl.` namespace, not to the position or the op token."""
    found = lint(TL_IN_HOST.replace("REPL", "torch.sum(out)"), "F1.4")
    assert [f.severity for f in found] == ["warn"]
    assert found[0].data["ops"] == ["torch.sum"]


# BUG-27: `_host_scopes` is `model.timed_scopes if model.entry else set(model.functions)`.
# The fallback abandons the precision the docstring is built on and scans *every*
# function in the file -- including the autograd `backward` the docstring names as the
# thing that must never be scanned. It triggers whenever ModelNew inherits its forward
# rather than defining one, which resolves `entry` to None.
INHERITED_FORWARD = '''
class Reference(nn.Module):
    """The original PyTorch model, left in the file. Nothing constructs it."""
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(n, n)

    def forward(self, x):
        return torch.matmul(self.fc(x), x)


class Base(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out


class ModelNew(Base):
    pass
'''

#: The sole edit: a forward that delegates to the one it would have inherited anyway.
EXPLICIT_FORWARD = INHERITED_FORWARD.replace(
    "class ModelNew(Base):\n    pass",
    "class ModelNew(Base):\n    def forward(self, x):\n        return super().forward(x)",
)

AUTOGRAD_BACKWARD = '''
class MyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.empty_like(x)
        work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
        return out

    @staticmethod
    def backward(ctx, g):
        return torch.matmul(g, g)


class Base(nn.Module):
    def forward(self, x):
        return MyFn.apply(x)


class ModelNew(Base):
    pass
'''


def test_inherited_forward_leaves_entry_unresolved():
    """Premise: `class ModelNew(Base): pass` is what puts `_host_scopes` on the
    all-functions fallback. The class is found; only its forward is inherited."""
    model = build_model(src(ELEMENTWISE_KERNEL + INHERITED_FORWARD), "<t>")
    assert model.model_class == "ModelNew"
    assert model.entry is None
    assert model.notes == ["ModelNew has no forward()"]


@pytest.mark.xfail(
    strict=True,
    reason="BUG-27: ModelNew inherits its forward, so entry is None and _host_scopes "
    "falls back to every function in the file. The dead `Reference` class -- the "
    "original PyTorch model, which nothing constructs -- is scanned and its "
    "torch.matmul reported at fail as what the timed forward computes",
)
def test_dead_reference_class_is_not_scanned_when_forward_is_inherited():
    assert lint(INHERITED_FORWARD, "F1.4") == []


@pytest.mark.xfail(
    strict=True,
    reason="BUG-27: the fallback scans the autograd `backward` that the _host_scopes "
    "docstring explicitly names as excluded ('backward/jvp/vmap never run under the "
    "benchmark's forward call, so a torch op there is not a fallback'). This re-opens "
    "the founding false positive of this whole audit history -- the pure-Triton Mish "
    "with a torch backward, p32_s6 -- for any solution that inherits its forward. "
    "Real sample: p14770_s6",
)
def test_autograd_backward_is_not_scanned_when_forward_is_inherited():
    assert lint(AUTOGRAD_BACKWARD, "F1.4") == []


def test_control_an_explicit_delegating_forward_is_silent():
    """The whole bug in one pair. This file differs from INHERITED_FORWARD only by a
    `def forward(self, x): return super().forward(x)` -- behaviourally identical code
    -- and it goes from `fail` to silent. The verdict tracks whether the entry point
    is spelled out, not what the model computes.
    """
    model = build_model(src(ELEMENTWISE_KERNEL + EXPLICIT_FORWARD), "<t>")
    assert model.entry == "ModelNew.forward"
    assert lint(EXPLICIT_FORWARD, "F1.4") == []
