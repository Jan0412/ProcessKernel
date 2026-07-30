"""Host-flow and kernel-body internals: aliasing, taint, reachability, propagation."""

from __future__ import annotations

from conftest import ELEMENTWISE_KERNEL, src

from checker.core.model import ModuleModel


def buf(model, name):
    return model.buffers[model.canonical(name)]


class TestAliasing:
    def test_bound_view_aliases_its_source(self, analyze):
        """`y = x.view(...)` is the same memory, so both names must resolve to one buffer."""
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        flat = x.view(-1)
        out = torch.empty_like(flat)
        add_kernel[(1,)](flat, y, out, flat.numel(), BLOCK=128)
        return out
"""
            )
        )
        assert m.canonical("ModelNew.forward::flat") == "ModelNew.forward::x"

    def test_plain_rebinding_aliases(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        z = x
        out = torch.empty_like(z)
        add_kernel[(1,)](z, y, out, z.numel(), BLOCK=128)
        return out
"""
            )
        )
        assert m.canonical("ModelNew.forward::z") == "ModelNew.forward::x"

    def test_noncontiguity_survives_a_bound_alias(self, analyze):
        """`xt = x.permute(...)` marks xt non-contiguous; `z = xt` must inherit that,
        otherwise the .contiguous() on z looks free when it really copies."""
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        xt = x.permute(1, 0)
        z = xt
        out = torch.empty_like(z)
        add_kernel[(1,)](z, y, out, z.numel(), BLOCK=128)
        return out
"""
            )
        )
        assert "ModelNew.forward::xt" in m.noncontiguous
        assert "ModelNew.forward::z" in m.noncontiguous

    def test_view_of_a_permute_stays_noncontiguous(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        xt = x.transpose(0, 1)
        flat = xt.reshape(-1)
        out = torch.empty_like(flat)
        add_kernel[(1,)](flat, y, out, flat.numel(), BLOCK=128)
        return out
"""
            )
        )
        assert "ModelNew.forward::flat" in m.noncontiguous


class TestReturnedNames:
    def test_subscript_of_a_buffer_flows_out(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out[0]
"""
            )
        )
        assert buf(m, "ModelNew.forward::out").returned

    def test_argument_of_a_helper_call_does_not_flow_out(self, analyze):
        """`return do_scale(tmp)` consumes tmp -- it does not return it. Treating it as
        returned would hide the very intermediate F2.1 exists to find."""
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
def do_scale(t):
    return t

class ModelNew(nn.Module):
    def forward(self, x, y):
        tmp = torch.empty_like(x)
        add_kernel[(1,)](x, y, tmp, x.numel(), BLOCK=128)
        return do_scale(tmp)
"""
            )
        )
        assert not buf(m, "ModelNew.forward::tmp").returned


class TestReachability:
    def test_follows_a_method_on_self(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def _run(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out

    def forward(self, x, y):
        return self._run(x, y)
"""
            )
        )
        assert "ModelNew._run" in m.reachable
        assert len(m.reachable_launches) == 1

    def test_locals_in_init_are_not_submodules(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        scale = 2.0
        self.scale = scale

    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out
"""
            )
        )
        assert m.nn_modules_in_init == {}
        assert len(m.reachable_launches) == 1


class TestInterproceduralPropagation:
    def test_a_caller_allocated_buffer_inherits_the_helpers_launch(self, analyze):
        """`fill(x, out)` writes into out. Without pushing the helper's param roles back
        to the caller, out looks like a buffer nobody ever wrote."""
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
def fill(a, b, dst):
    add_kernel[(1,)](a, b, dst, a.numel(), BLOCK=128)

class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        fill(x, y, out)
        return out
"""
            )
        )
        out = buf(m, "ModelNew.forward::out")
        assert out.stored_by == [0]  # the launch inside fill()
        assert buf(m, "ModelNew.forward::x").loaded_by == [0]

    def test_non_tensor_arguments_are_skipped(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
def scale(a, b, dst, factor):
    add_kernel[(1,)](a, b, dst, a.numel(), BLOCK=128)

class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        scale(x, y, out, 2.0)
        return out
"""
            )
        )
        assert buf(m, "ModelNew.forward::out").stored_by == [0]


class TestKernelBodyTaint:
    def test_pointer_arithmetic_through_a_local(self, analyze):
        """row = x_ptr + pid * stride; tl.load(row + offs) still marks x_ptr as loaded."""
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, out_ptr, stride, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    row = x_ptr + pid * stride
    dst = out_ptr + pid * stride
    v = tl.load(row + offs)
    tl.store(dst + offs, v * 2.0)
"""
            )
        )
        params = m.kernels["k"].params
        assert params["x_ptr"].loaded and not params["x_ptr"].stored
        assert params["out_ptr"].stored
        assert not params["stride"].is_pointer  # offset arithmetic, not a pointer

    def test_triton_language_spelled_out(self, analyze):
        m = analyze(
            src(
                """
import triton.language

@triton.jit
def k(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = triton.language.arange(0, BLOCK)
    v = triton.language.load(x_ptr + offs)
    triton.language.store(out_ptr + offs, v + 1.0)
"""
            )
        )
        assert m.kernels["k"].params["x_ptr"].loaded
        assert m.kernels["k"].params["out_ptr"].stored

    def test_a_rebound_local_loses_its_pointer_taint(self, analyze):
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = x_ptr + offs
    p = tl.sum(tl.load(p), axis=0)
    tl.store(out_ptr + offs, p)
"""
            )
        )
        assert m.kernels["k"].params["x_ptr"].loaded

    def test_storing_a_load_directly_is_a_copy(self, analyze):
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)
"""
            )
        )
        assert m.kernels["k"].kind == "copy"

    def test_copy_through_a_plain_rebinding(self, analyze):
        """v = tl.load(...); w = v; tl.store(out, w) -- still a pure copy."""
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)
    w = v
    tl.store(out_ptr + offs, w, mask=offs < n)
"""
            )
        )
        assert m.kernels["k"].kind == "copy"

    def test_accumulator_in_a_for_loop_is_a_reduction(self, analyze):
        """No tl.sum, but `acc += ...` inside a loop is a reduction all the same."""
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, n, BLOCK):
        acc += tl.load(x_ptr + i + tl.arange(0, BLOCK))
    tl.store(out_ptr + tl.program_id(0), acc)
"""
            )
        )
        assert m.kernels["k"].kind == "reduction"

    def test_accumulator_in_a_while_loop_is_a_reduction(self, analyze):
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    i = 0
    while i < n:
        acc += tl.load(x_ptr + i + tl.arange(0, BLOCK))
        i += BLOCK
    tl.store(out_ptr + tl.program_id(0), acc)
"""
            )
        )
        assert m.kernels["k"].kind == "reduction"

    def test_a_kernel_that_stores_nothing_is_unknown(self, analyze):
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)
    helper(v)
"""
            )
        )
        assert m.kernels["k"].kind == "unknown"
        assert m.kernels["k"].outputs() == []


class TestParsing:
    def test_kernel_nested_in_a_class(self, analyze):
        """Unusual, but legal -- and it must not be mistaken for a host method."""
        m = analyze(
            src(
                """
class ModelNew(nn.Module):
    @triton.jit
    def k(x_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) * 2.0)

    def forward(self, x):
        out = torch.empty_like(x)
        ModelNew.k[(1,)](x, out, BLOCK=128)
        return out
"""
            )
        )
        assert "k" in m.kernels
        assert "ModelNew.k" not in m.functions

    def test_entry_class_without_a_forward(self, analyze):
        m = analyze(src("class ModelNew(nn.Module):\n    pass\n"))
        assert m.model_class == "ModelNew"
        assert m.entry is None
        assert "ModelNew has no forward()" in m.notes

    def test_no_entry_class_at_all(self, analyze):
        m = analyze(src(ELEMENTWISE_KERNEL))
        assert "no entry-point class found" in m.notes


class TestModuleModel:
    def test_buffer_resolves_through_aliases(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        flat = x.view(-1)
        out = torch.empty_like(flat)
        add_kernel[(1,)](flat, y, out, flat.numel(), BLOCK=128)
        return out
"""
            )
        )
        assert m.buffer("ModelNew.forward::flat") is m.buffer("ModelNew.forward::x")
        assert m.buffer("ModelNew.forward::nonexistent") is None

    def test_canonical_survives_an_alias_cycle(self):
        model = ModuleModel()
        model.aliases = {"a": "b", "b": "a"}
        assert model.canonical("a") in ("a", "b")  # terminates rather than looping
