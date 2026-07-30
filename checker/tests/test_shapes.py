"""shapes.py -- static shape inference, which is what turns a finding into bytes."""

from __future__ import annotations

import ast

import pytest
from conftest import ELEMENTWISE_KERNEL, src

from checker.shapes import (
    _const_int,
    _const_shape,
    _dtype_of,
    _innermost_call,
    _nbytes,
    _shape_from_call,
    reference_input_shapes,
    shapes_from_source,
)

SHAPES = [((64, 64), "float32"), ((64, 64), "float32")]


def expr(code: str) -> ast.expr:
    return ast.parse(code, mode="eval").body


def call(code: str) -> ast.Call:
    node = expr(code)
    assert isinstance(node, ast.Call)
    return node


class TestConstInt:
    @pytest.mark.parametrize(
        "code,expected",
        [
            ("7", 7),
            ("B", 4),
            ("B + 2", 6),
            ("B - 1", 3),
            ("B * H", 32),
            ("H // 2", 4),
            ("H / 2", 4),  # a shape is an int: Div floors, like FloorDiv
        ],
    )
    def test_evaluates_integer_arithmetic(self, code, expected):
        assert _const_int(expr(code), {"B": 4, "H": 8}) == expected

    @pytest.mark.parametrize(
        "code",
        [
            "unknown",  # name not in the environment
            "unknown + 1",  # ...propagates through the arithmetic
            "1 + unknown",
            "B % 2",  # an operator we do not model
            "B // 0",  # ZeroDivisionError, not a crash
            "'four'",  # not an int
            "f(B)",  # not arithmetic at all
        ],
    )
    def test_unresolvable_stays_none(self, code):
        assert _const_int(expr(code), {"B": 4}) is None

    def test_const_shape_is_all_or_nothing(self):
        assert _const_shape([expr("2"), expr("B")], {"B": 4}) == (2, 4)
        assert _const_shape([expr("2"), expr("unknown")], {}) is None


class TestDtypeOf:
    def test_defaults_to_float32(self):
        assert _dtype_of(call("torch.rand([4])")) == "float32"

    def test_reads_the_dtype_keyword(self):
        assert _dtype_of(call("torch.empty(4, dtype=torch.float16)")) == "float16"

    def test_randint_is_int64(self):
        assert _dtype_of(call("torch.randint(0, 9, [4])")) == "int64"

    def test_explicit_default(self):
        assert _dtype_of(call("torch.empty(4)"), "bfloat16") == "bfloat16"

    def test_undottable_dtype_falls_back(self):
        assert _dtype_of(call("torch.empty(4, dtype=dtypes[0])")) == "float32"


class TestShapeFromCall:
    @pytest.mark.parametrize(
        "code,expected",
        [
            ("torch.rand([4, 8])", (4, 8)),  # list literal
            ("torch.rand((4, 8))", (4, 8)),  # tuple literal
            ("torch.rand(4, 8)", (4, 8)),  # varargs
            ("torch.empty(size=[2, 3])", (2, 3)),  # size= keyword
            ("torch.randint(0, 9, [4, 4])", (4, 4)),  # low/high come first
            ("torch.rand(B, 8)", (4, 8)),  # resolved against the environment
        ],
    )
    def test_recognised_forms(self, code, expected):
        assert _shape_from_call(call(code), {"B": 4}) == expected

    @pytest.mark.parametrize(
        "code",
        [
            "torch.matmul(a, b)",  # not an allocation
            "torch.rand(x.shape)",  # an expression we cannot evaluate
            "torch.rand(n)",  # n is not bound
        ],
    )
    def test_unresolvable_stays_none(self, code):
        assert _shape_from_call(call(code), {}) is None

    def test_nbytes_uses_the_dtype_itemsize(self):
        assert _nbytes((4, 8), "float32") == 128
        assert _nbytes((4, 8), "float16") == 64
        assert _nbytes((4, 8), "not_a_dtype") == 128  # unknown dtypes assume 4 bytes


class TestInnermostCall:
    def test_unwraps_method_chains(self):
        """torch.rand([4, 4]).cuda() -- the shape lives on the rand call."""
        found = _innermost_call(expr("torch.rand([4, 4]).cuda()"))
        assert _shape_from_call(found, {}) == (4, 4)

    def test_falls_back_to_the_outermost_call(self):
        assert _innermost_call(expr("make_input(1).cuda()")).func.id == "make_input"

    def test_returns_none_without_a_call(self):
        assert _innermost_call(expr("x")) is None


class TestGetInputs:
    def test_reads_shapes_and_dtypes(self):
        shapes = shapes_from_source(
            "import torch\n"
            "def get_inputs():\n"
            "    return [torch.rand([4, 8]), torch.randint(0, 9, [2], dtype=torch.int32)]\n"
        )
        assert shapes == [((4, 8), "float32"), ((2,), "int32")]

    def test_unresolvable_entries_become_none(self):
        """A shape we cannot recover is None -- never a guess."""
        shapes = shapes_from_source(
            "import torch\n"
            "def get_inputs():\n"
            "    return [x, torch.rand(n), torch.rand([4])]\n"
        )
        assert shapes == [None, None, ((4,), "float32")]

    def test_return_that_is_not_a_sequence(self):
        assert shapes_from_source("def get_inputs():\n    return make_them()\n") == []

    def test_no_get_inputs(self):
        assert shapes_from_source("import torch\n") == []

    def test_unparseable_source(self):
        assert shapes_from_source("def get_inputs(:\n") == []


class TestReferenceInputShapes:
    def test_reads_the_kernelbench_reference(self, fake_kernelbench):
        assert reference_input_shapes(1, 2) == [((4, 8), "float32"), ((4, 8), "float32")]

    def test_missing_reference(self, fake_kernelbench):
        assert reference_input_shapes(1, 999) == []


def buf(model, name):
    return model.buffers[model.canonical(name)]


class TestNameBinding:
    """`env` is populated from the shape-query idioms, then sizes the allocations."""

    def test_tuple_unpacked_shape(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        B, C, H, W = x.shape
        out = torch.empty((B, C, H, W))
        add_kernel[(1,)](x, y, out, B * C * H * W, BLOCK=128)
        return out
"""
            ),
            [((2, 3, 4, 5), "float32"), ((2, 3, 4, 5), "float32")],
        )
        assert buf(m, "ModelNew.forward::out").shape == (2, 3, 4, 5)
        assert buf(m, "ModelNew.forward::out").nbytes == 2 * 3 * 4 * 5 * 4

    def test_numel(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        n = x.numel()
        out = torch.empty(n)
        add_kernel[(1,)](x, y, out, n, BLOCK=128)
        return out
"""
            ),
            SHAPES,
        )
        assert buf(m, "ModelNew.forward::out").shape == (64 * 64,)

    def test_size_and_subscripted_shape(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        B = x.size(0)
        C = x.shape[1]
        out = torch.empty(B, C)
        add_kernel[(1,)](x, y, out, B * C, BLOCK=128)
        return out
"""
            ),
            [((8, 16), "float32"), ((8, 16), "float32")],
        )
        assert buf(m, "ModelNew.forward::out").shape == (8, 16)

    def test_out_of_range_index_is_ignored(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        D = x.shape[7]
        out = torch.empty(D)
        add_kernel[(1,)](x, y, out, D, BLOCK=128)
        return out
"""
            ),
            [((8, 16), "float32"), ((8, 16), "float32")],
        )
        assert buf(m, "ModelNew.forward::out").shape is None


class TestAllocationSizing:
    def test_empty_like_inherits_shape_and_dtype(self, analyze):
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
            [((4, 4), "float16"), ((4, 4), "float16")],
        )
        out = buf(m, "ModelNew.forward::out")
        assert out.shape == (4, 4) and out.dtype == "float16"
        assert out.nbytes == 4 * 4 * 2

    def test_empty_like_with_a_dtype_override(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x, dtype=torch.float64)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out
"""
            ),
            [((4, 4), "float32"), ((4, 4), "float32")],
        )
        assert buf(m, "ModelNew.forward::out").nbytes == 4 * 4 * 8

    def test_allocation_inherits_the_input_dtype(self, analyze):
        """No dtype= on the allocation: it defaults to the dtype of the inputs."""
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        n = x.numel()
        out = torch.empty(n)
        add_kernel[(1,)](x, y, out, n, BLOCK=128)
        return out
"""
            ),
            [((4, 4), "bfloat16"), ((4, 4), "bfloat16")],
        )
        assert buf(m, "ModelNew.forward::out").dtype == "bfloat16"

    def test_like_of_an_unknown_tensor_stays_unsized(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(mystery)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out
"""
            )
        )
        assert buf(m, "ModelNew.forward::out").nbytes is None


class TestInterproceduralShapes:
    def test_helper_parameters_inherit_the_callers_shapes(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
def do_add(a, b):
    out = torch.empty_like(a)
    add_kernel[(1,)](a, b, out, a.numel(), BLOCK=128)
    return out

class ModelNew(nn.Module):
    def forward(self, x, y):
        return do_add(x, y)
"""
            ),
            SHAPES,
        )
        assert buf(m, "do_add::a").shape == (64, 64)
        assert buf(m, "do_add::out").nbytes == 64 * 64 * 4

    def test_a_helpers_return_value_carries_its_shape_back(self, analyze):
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
def do_add(a, b):
    out = torch.empty_like(a)
    add_kernel[(1,)](a, b, out, a.numel(), BLOCK=128)
    return out

class ModelNew(nn.Module):
    def forward(self, x, y):
        tmp = do_add(x, y)
        return tmp
"""
            ),
            SHAPES,
        )
        assert buf(m, "ModelNew.forward::tmp").nbytes == 64 * 64 * 4


class TestNoShapes:
    def test_get_inputs_wins_over_the_fallback(self, analyze):
        """The file's own get_inputs() is the truth when it has one."""
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
    return [torch.rand([2, 2]), torch.rand([2, 2])]
"""
            ),
            SHAPES,  # a (64, 64) fallback, which must be ignored
        )
        assert buf(m, "ModelNew.forward::out").nbytes == 2 * 2 * 4

    def test_no_entry_point(self, analyze):
        """A truncated file with no ModelNew: inference stops, nothing raises."""
        m = analyze(src(ELEMENTWISE_KERNEL), SHAPES)
        assert m.parse_status == "ok"
        assert m.entry is None

    def test_dtype_defaults_to_float32_without_any_inputs(self, analyze):
        """Nothing to infer a dtype from -- fall back to float32 rather than refuse."""
        m = analyze(
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty(4, 8)
        add_kernel[(1,)](x, y, out, 32, BLOCK=128)
        return out
"""
            )
        )
        out = buf(m, "ModelNew.forward::out")
        assert out.dtype == "float32"
        assert out.nbytes == 4 * 8 * 4
