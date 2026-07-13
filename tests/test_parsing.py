"""Infrastructure: decorator forms, entry-point resolution, launches, roles, shapes."""

from __future__ import annotations

from conftest import ELEMENTWISE_KERNEL, src

from triton_lint.model import parse_kernel_filename, staged_kernel_filename


class TestKernelDetection:
    def test_plain_jit(self, analyze):
        m = analyze(src(ELEMENTWISE_KERNEL))
        assert set(m.kernels) == {"add_kernel"}

    def test_autotune_stacked_above_jit(self, analyze):
        m = analyze(
            src(
                """
@triton.autotune(configs=[triton.Config({'BLOCK': 128})], key=['n'])
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) * 2.0)
"""
            )
        )
        assert "k" in m.kernels
        assert m.kernels["k"].has_autotune

    def test_bare_jit_from_import(self, analyze):
        m = analyze(
            "from triton import jit\nimport triton.language as tl\n"
            "@jit\ndef k(x_ptr, out_ptr):\n    tl.store(out_ptr, tl.load(x_ptr) + 1)\n"
        )
        assert "k" in m.kernels

    def test_jit_call_form(self, analyze):
        m = analyze(
            src(
                """
@triton.jit()
def k(x_ptr, out_ptr):
    tl.store(out_ptr, tl.load(x_ptr) + 1)
"""
            )
        )
        assert "k" in m.kernels


class TestEntryPoint:
    def test_model_new(self, analyze):
        m = analyze(src("class ModelNew(nn.Module):\n    def forward(self, x):\n        return x\n"))
        assert m.entry == "ModelNew.forward"

    def test_model_alias(self, analyze):
        m = analyze(
            src(
                """
class CustomLayer(nn.Module):
    def forward(self, x):
        return x

Model = CustomLayer
"""
            )
        )
        assert m.model_class == "CustomLayer"
        assert m.entry == "CustomLayer.forward"

    def test_sole_nn_module_fallback(self, analyze):
        m = analyze(src("class Whatever(nn.Module):\n    def forward(self, x):\n        return x\n"))
        assert m.entry == "Whatever.forward"


class TestLaunches:
    def test_launch_via_helper_two_hops(self, analyze):
        """A launch two calls deep from forward is still reachable."""
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
def inner(x, y):
    out = torch.empty_like(x)
    add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
    return out

def outer(x, y):
    return inner(x, y)

class ModelNew(nn.Module):
    def forward(self, x, y):
        return outer(x, y)
"""
            )
        )
        assert len(m.reachable_launches) == 1
        assert "inner" in m.reachable

    def test_grid_as_lambda(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK']),)
        add_kernel[grid](x, y, out, x.numel(), BLOCK=128)
        return out
"""
            )
        )
        assert len(m.reachable_launches) == 1

    def test_launch_in_loop_records_depth(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            add_kernel[(1,)](x[i], y[i], out[i], x[i].numel(), BLOCK=128)
        return out
"""
            )
        )
        assert m.reachable_launches[0].loop_depth == 1

    def test_submodule_class_is_reachable(self, analyze):
        """self.norm = LayerNormTriton(...) -- the launch lives in the submodule."""
        m = analyze(
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
            )
        )
        assert len(m.reachable_launches) == 1

    def test_autograd_function_is_reachable(self, analyze):
        """MyFn.apply(x) -- the launch lives in MyFn.forward."""
        m = analyze(
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
            )
        )
        assert len(m.reachable_launches) == 1


class TestParamRoles:
    def test_store_and_load_targets(self, analyze):
        m = analyze(src(ELEMENTWISE_KERNEL))
        k = m.kernels["add_kernel"]
        assert k.outputs() == ["out_ptr"]
        assert set(k.inputs()) == {"x_ptr", "y_ptr"}

    def test_shape_scalars_are_not_pointers(self, analyze):
        """H and W appear inside the load's offset arithmetic but are not tensors."""
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, out_ptr, N, C, H, W, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    base = ((pid * C + 0) * H) * W
    v = tl.load(x_ptr + base + c * (H * W))
    tl.store(out_ptr + pid, tl.sum(v, axis=0))
"""
            )
        )
        k = m.kernels["k"]
        for scalar in ("N", "C", "H", "W", "BLOCK"):
            assert not k.params[scalar].is_pointer, f"{scalar} wrongly seen as a pointer"
        assert k.params["x_ptr"].loaded
        assert k.params["out_ptr"].stored

    def test_atomic_marks_both(self, analyze):
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.atomic_add(out_ptr + offs, tl.load(x_ptr + offs))
"""
            )
        )
        role = m.kernels["k"].params["out_ptr"]
        assert role.atomic and role.stored and role.loaded


class TestKernelKind:
    def test_elementwise(self, analyze):
        assert analyze(src(ELEMENTWISE_KERNEL)).kernels["add_kernel"].kind == "elementwise"

    def test_reduction_via_tl_sum(self, analyze):
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.program_id(0), tl.sum(tl.load(x_ptr + offs), axis=0))
"""
            )
        )
        assert m.kernels["k"].kind == "reduction"

    def test_matmul_via_tl_dot(self, analyze):
        m = analyze(
            src(
                """
@triton.jit
def k(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(c_ptr + offs, tl.dot(a, b))
"""
            )
        )
        assert m.kernels["k"].kind == "matmul"

    def test_copy(self, analyze):
        m = analyze(
            src(
                """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < n)
    tl.store(out_ptr + offs, v, mask=offs < n)
"""
            )
        )
        assert m.kernels["k"].kind == "copy"


class TestDegradedInputs:
    def test_empty_file(self, analyze):
        assert analyze("").parse_status == "empty"

    def test_syntax_error(self, analyze):
        assert analyze("def f(:\n  pass").parse_status == "syntax_error"

    def test_truncated_kernel_only(self, analyze):
        """Some generations are cut off mid-file: kernel, no ModelNew."""
        m = analyze(src(ELEMENTWISE_KERNEL))
        assert m.parse_status == "ok"
        assert m.entry is None
        assert "add_kernel" in m.kernels


class TestFilenames:
    def test_roundtrip(self):
        name = staged_kernel_filename(5, 123, 7)
        assert name == "level_5_problem_123_sample_7_kernel.py"
        assert parse_kernel_filename(name) == (5, 123, 7)

    def test_rejects_other_files(self):
        assert parse_kernel_filename("eval_results.json") is None
        assert parse_kernel_filename("generation_config.yaml") is None


class TestShapes:
    def test_get_inputs(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out

def get_inputs():
    return [torch.rand([4, 8]), torch.rand([4, 8])]
"""
            )
        )
        assert m.input_shapes[0] == ((4, 8), "float32")
        out = m.buffers[m.canonical("ModelNew.forward::out")]
        assert out.nbytes == 4 * 8 * 4

    def test_fallback_shapes_used_when_get_inputs_missing(self, analyze):
        """Most generations drop get_inputs(); we fall back to the reference's."""
        m = analyze(
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
            [((2, 2), "float32"), ((2, 2), "float32")],
        )
        out = m.buffers[m.canonical("ModelNew.forward::out")]
        assert out.nbytes == 2 * 2 * 4
