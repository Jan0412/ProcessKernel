"""F2.1 dead_intermediate."""

from __future__ import annotations

from conftest import src
from helpers import lint, lint_raw

from checker import build_model
from checker.lint.checks.family2 import f2_1_dead_intermediate

from ._fixtures import NBYTES, REDUCE_KERNEL, SHAPES, THREE_LAUNCHES, TWO_ELEMENTWISE


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


# ---------------------------------------------------------------------------
# Regression tests for former linter bugs, now fixed (history in tests/BUGS.md).
# ---------------------------------------------------------------------------

HELPER_PRODUCED_REUSED = '''
def t_relu(x):
    out = torch.empty_like(x)
    work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
    return out


def t_scale(x):
    out = torch.empty_like(x)
    work_kernel[(1,)](x, out, x.numel(), BLOCK=1024)
    return out


class ModelNew(nn.Module):
    def forward(self, x):
        enc = t_relu(x)
        pooled = t_scale(enc)
        return torch.cat([pooled, enc], dim=1)
'''


def test_helper_produced_tensor_reused_by_host_is_not_dead():
    findings = lint(HELPER_PRODUCED_REUSED, "F2.1")
    assert not any("enc" in f.data["intermediates"] for f in findings)


#: A max-pool: the reduction is carried across a loop by re-assignment, not by ``+=``.
LOOP_CARRIED_POOL = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def pool_kernel(x_ptr, out_ptr, n, K, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    for k in range(K):
        v = tl.load(x_ptr + offs * K + k, mask=mask, other=-float("inf"))
        acc = tl.maximum(acc, v)
    tl.store(out_ptr + offs, acc, mask=mask)

@triton.jit
def scale_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) * 2.0, mask=offs < n)

class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        pooled = torch.empty_like(x)
        pool_kernel[(1,)](x, pooled, n, 4, BLOCK=128)
        out = torch.empty_like(pooled)
        scale_kernel[(1,)](pooled, out, n, BLOCK=128)
        return out
'''


def test_augassign_accumulator_is_a_reduction():
    """The control for BUG-13: the `+=` spelling *is* recognised."""
    model = build_model(LOOP_CARRIED_POOL.replace("acc = tl.maximum(acc, v)", "acc += v"), "<t>")
    assert model.kernels["pool_kernel"].kind == "reduction"


def test_loop_carried_reduction_is_not_fusible_into_its_consumer():
    assert build_model(LOOP_CARRIED_POOL, "<t>").kernels["pool_kernel"].kind == "reduction"
    findings = lint_raw(LOOP_CARRIED_POOL, "F2.1")
    assert len(findings) == 1
    assert findings[0].data["fusible"] is False
    assert "Fuse" not in findings[0].message


#: A diamond, the shape p2084_s2 has: `add_kernel` (launch 3) consumes two dead
#: intermediates -- `b` (from launch 1) and `c` (from launch 2, itself fed by `a`
#: from launch 0). All four launches are one connected component, so the merge
#: contract ("k1 -> t1 -> k2 -> t2 -> k3 becomes ONE finding") demands a single
#: finding over {a, b, c}.
DIAMOND = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def act_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(out_ptr + o, tl.sigmoid(tl.load(x_ptr + o, mask=m)), mask=m)

@triton.jit
def scale_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(out_ptr + o, tl.load(x_ptr + o, mask=m) * 2.0, mask=m)

@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(out_ptr + o, tl.load(a_ptr + o, mask=m) + tl.load(b_ptr + o, mask=m), mask=m)

class ModelNew(nn.Module):
    def forward(self, x, y):
        n = x.numel()
        a = torch.empty_like(x)
        act_kernel[(1,)](x, a, n, BLOCK=128)        # launch 0: -> a
        b = torch.empty_like(y)
        act_kernel[(1,)](y, b, n, BLOCK=128)        # launch 1: -> b
        c = torch.empty_like(x)
        scale_kernel[(1,)](a, c, n, BLOCK=128)      # launch 2: a -> c
        out = torch.empty_like(x)
        add_kernel[(1,)](b, c, out, n, BLOCK=128)   # launch 3: (b, c) -> out
        return out
'''

#: The control: two producers feeding ONE consumer. Here the second intermediate
#: attaches to the first chain *via the shared consumer* on the same pass, so the
#: merge succeeds -- pinning the boundary as being about processing order, not shape.
DIAMOND_MERGES = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def act_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(out_ptr + o, tl.sigmoid(tl.load(x_ptr + o, mask=m)), mask=m)

@triton.jit
def scale_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(out_ptr + o, tl.load(x_ptr + o, mask=m) * 2.0, mask=m)

@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(out_ptr + o, tl.load(a_ptr + o, mask=m) + tl.load(b_ptr + o, mask=m), mask=m)

class ModelNew(nn.Module):
    def forward(self, x, y):
        n = x.numel()
        a = torch.empty_like(x)
        act_kernel[(1,)](x, a, n, BLOCK=128)
        b = torch.empty_like(y)
        scale_kernel[(1,)](y, b, n, BLOCK=128)
        out = torch.empty_like(x)
        add_kernel[(1,)](a, b, out, n, BLOCK=128)
        return out
'''


def test_two_producers_one_consumer_merge_into_one_finding():
    """Control for BUG-17: when the bridging intermediate is processed *before* a
    separate chain forms, `_build_chains` merges it correctly -- one finding."""
    findings = lint_raw(DIAMOND_MERGES, "F2.1")
    assert len(findings) == 1
    assert set(findings[0].data["intermediates"]) == {"a", "b"}


def test_diamond_shares_no_kernel_across_findings():
    findings = lint_raw(DIAMOND, "F2.1")
    # The four launches are one connected component -> exactly one finding.
    assert len(findings) == 1


# ---------------------------------------------------------------------------
# Only reachable launches count: a round-trip that lives in dead code is not a cost.
# ---------------------------------------------------------------------------


def test_dead_intermediate_in_unreachable_code_is_skipped(check):
    """`_is_dead_intermediate` runs over every buffer, but the producer and consumer
    must be *reachable* launches. A round-trip confined to a helper the entry point
    never calls resolves to no reachable producer, so it is skipped -- exercising the
    `producer is None or not consumers` branch.
    """
    found = check(
        "F2.1",
        src(
            TWO_ELEMENTWISE
            + """
def unused(x):
    n = x.numel()
    tmp = torch.empty_like(x)
    exp_kernel[(1,)](x, tmp, n, BLOCK=128)
    out = torch.empty_like(x)
    scale_kernel[(1,)](tmp, out, n, BLOCK=128)
    return out

class ModelNew(nn.Module):
    def forward(self, x):
        n = x.numel()
        a = torch.empty_like(x)
        exp_kernel[(1,)](x, a, n, BLOCK=128)
        b = torch.empty_like(x)
        scale_kernel[(1,)](x, b, n, BLOCK=128)
        return a + b
""",
        ),
        SHAPES,
    )
    # forward has the two reachable launches (so the < 2 early-out is not taken), but its
    # buffers are both returned; the only round-trip, `tmp`, is in unreachable `unused`.
    assert found == []


def test_chain_kernel_absent_from_the_model_is_skipped():
    """`_finding_for` resolves each launch's kernel through `model.kernels`; after a
    normal build every reachable launch has one. The `pk is None or ck is None` guard
    covers a broken mapping -- exercise it directly and confirm the finding degrades
    gracefully (no fusible pair, so it is reported as materialisation cost).
    """
    model = build_model(
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
        "<t>",
        SHAPES,
    )
    assert len(f2_1_dead_intermediate.DeadIntermediate().run(model)) == 1  # the intermediate is real
    del model.kernels["exp_kernel"]
    del model.kernels["scale_kernel"]
    findings = f2_1_dead_intermediate.DeadIntermediate().run(model)
    assert len(findings) == 1
    assert findings[0].data["kernels"] == []  # every pair dropped by the guard
