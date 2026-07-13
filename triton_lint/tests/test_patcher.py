"""Knob discovery and config patching.

The fixtures below are the shapes that actually occur in runs/ (see the survey in
autotune/knobs.py), plus the edge cases that would corrupt a kernel silently if we got
them wrong -- a trailing comma at the launch, a block size computed at runtime, a
constexpr that is a real dimension rather than a tile size.
"""

from __future__ import annotations

import ast

import pytest

from autotune.grids import build_grid
from autotune.knobs import analyze, gaming_report
from autotune.patcher import Unpatchable, patch_source

# The dominant shape: 163k of the 175k kernels in runs/ look like this.
ASSIGN_META = '''
import torch, triton, triton.language as tl

@triton.jit
def relu_kernel(x_ptr, o_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)

def triton_relu(x):
    o = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    relu_kernel[grid](x, o, n, BLOCK_SIZE=BLOCK_SIZE)
    return o

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return triton_relu(x)
'''

LAUNCH_LITERAL = '''
import triton, triton.language as tl

@triton.jit
def k(p, n, BLOCK_SIZE: tl.constexpr):
    tl.store(p + tl.arange(0, BLOCK_SIZE), 0.0)

def run(p, n):
    grid = (1,)
    k[grid](p, n, BLOCK_SIZE=512)
'''

# The grid closes over the Python variable instead of reading meta[...]. Patching the
# assignment keeps them in sync for free; patching the launch kwarg alone would not.
CLOSURE_GRID = '''
import triton, triton.language as tl

@triton.jit
def k(p, n, BLOCK_SIZE: tl.constexpr):
    tl.store(p + tl.arange(0, BLOCK_SIZE), 0.0)

def run(p, n):
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n, BLOCK_SIZE),)
    k[grid](p, n, BLOCK_SIZE=BLOCK_SIZE)
'''

TRAILING_COMMA = '''
import triton, triton.language as tl

@triton.jit
def k(p, n, BLOCK_SIZE: tl.constexpr):
    tl.store(p + tl.arange(0, BLOCK_SIZE), 0.0)

def run(p, n):
    BLOCK_SIZE = 64
    k[(1,)](
        p,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
'''

# A trailing comment inside the launch. Scanning backwards from ')' for a comma lands on the
# comment text and produces "BLOCK_SIZE=1, , num_warps=1)". Found in the real corpus.
TRAILING_COMMENT = '''
import triton, triton.language as tl

@triton.jit
def k(p, n, BLOCK_SIZE: tl.constexpr):
    tl.store(p + tl.arange(0, BLOCK_SIZE), 0.0)

def run(p, n):
    grid = (n,)
    k[grid](
        p,
        n,
        BLOCK_SIZE=1,               # each program processes one element
    )
'''

HAS_NUM_WARPS = '''
import triton, triton.language as tl

@triton.jit
def k(p, n, BLOCK_SIZE: tl.constexpr):
    tl.store(p + tl.arange(0, BLOCK_SIZE), 0.0)

def run(p, n):
    BLOCK_SIZE = 64
    k[(1,)](p, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
'''

DYNAMIC_BLOCK = '''
import triton, triton.language as tl

@triton.jit
def k(p, n, BLOCK_SIZE: tl.constexpr):
    tl.store(p + tl.arange(0, BLOCK_SIZE), 0.0)

def run(p, n):
    BLOCK_SIZE = triton.next_power_of_2(n)
    k[(1,)](p, n, BLOCK_SIZE=BLOCK_SIZE)
'''

# HEAD_DIM is a real dimension, not a tile size: changing it makes the kernel wrong.
SEMANTIC_CONSTEXPR = '''
import triton, triton.language as tl

@triton.jit
def k(p, n, HEAD_DIM: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    tl.store(p + tl.arange(0, BLOCK_SIZE), 0.0)

def run(p, n, head_dim):
    BLOCK_SIZE = 64
    k[(1,)](p, n, HEAD_DIM=head_dim, IS_CAUSAL=True, BLOCK_SIZE=BLOCK_SIZE)
'''

TILED = '''
import triton, triton.language as tl

@triton.jit
def matmul_kernel(a, b, c, M, N, K,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        acc += 1.0
    tl.store(c, acc)

def run(a, b, c, M, N, K):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M)
'''


def _knob_names(src):
    return sorted(k.name for k in analyze(src).knobs)


class TestDiscovery:
    def test_assignment_bound_knob(self):
        rep = analyze(ASSIGN_META)
        assert _knob_names(ASSIGN_META) == ["BLOCK_SIZE"]
        knob = rep.knob("BLOCK_SIZE")
        assert knob.kind == "assign" and knob.current == 128
        assert rep.ndim_class == "1d" and rep.n_jit_kernels == 1 and rep.n_launches == 1

    def test_launch_literal(self):
        rep = analyze(LAUNCH_LITERAL)
        assert rep.knob("BLOCK_SIZE").kind == "launch_literal"
        assert rep.knob("BLOCK_SIZE").current == 512

    def test_runtime_computed_block_is_excluded(self):
        rep = analyze(DYNAMIC_BLOCK)
        assert rep.knobs == []
        assert rep.excluded and rep.excluded[0][0] == "BLOCK_SIZE"

    def test_semantic_constexpr_is_not_a_knob(self):
        # HEAD_DIM and IS_CAUSAL must not be swept; only the tile size is ours.
        assert _knob_names(SEMANTIC_CONSTEXPR) == ["BLOCK_SIZE"]

    def test_tiled_kernel(self):
        rep = analyze(TILED)
        assert _knob_names(TILED) == [
            "BLOCK_SIZE_K", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "GROUP_SIZE_M",
        ]
        assert rep.ndim_class == "tiled"
        assert rep.has_loop is True  # -> num_stages is meaningful

    def test_syntax_error_is_reported_not_raised(self):
        rep = analyze("def broken(:\n  pass")
        assert rep.parse_error is not None and rep.knobs == []


class TestPatching:
    def test_identity_config_is_byte_identical(self):
        # Config 0 is the denominator of every tuning gain; it must be the same file.
        for src in (ASSIGN_META, TILED, LAUNCH_LITERAL, HAS_NUM_WARPS):
            assert patch_source(src, {}) == src

    def test_patches_the_assignment_not_the_use(self):
        out = patch_source(ASSIGN_META, {"BLOCK_SIZE": 1024})
        assert "BLOCK_SIZE = 1024" in out
        # the launch and the grid still refer to the name, so they follow automatically
        assert 'meta["BLOCK_SIZE"]' in out
        assert "BLOCK_SIZE=BLOCK_SIZE)" in out
        assert analyze(out).knob("BLOCK_SIZE").current == 1024

    def test_closure_grid_stays_consistent(self):
        out = patch_source(CLOSURE_GRID, {"BLOCK_SIZE": 1024})
        assert "BLOCK_SIZE = 1024" in out
        assert "triton.cdiv(n, BLOCK_SIZE)" in out  # picks up the new value by closure

    def test_patches_launch_literal(self):
        out = patch_source(LAUNCH_LITERAL, {"BLOCK_SIZE": 64})
        assert "BLOCK_SIZE=64" in out

    def test_injects_num_warps(self):
        out = patch_source(ASSIGN_META, {"BLOCK_SIZE": 256, "num_warps": 8})
        assert "num_warps=8" in out
        ast.parse(out)

    def test_injects_num_warps_after_trailing_comma(self):
        # The bug this guards: "BLOCK_SIZE=BLOCK_SIZE,\n, num_warps=8)" -- a syntax error.
        out = patch_source(TRAILING_COMMA, {"num_warps": 8})
        ast.parse(out)
        assert ",," not in out.replace(" ", "").replace("\n", "")
        assert "num_warps=8" in out

    def test_replaces_existing_num_warps(self):
        out = patch_source(HAS_NUM_WARPS, {"num_warps": 8})
        assert "num_warps=8" in out and "num_warps=2" not in out

    def test_num_warps_and_num_stages_do_not_splice_into_each_other(self):
        # Both want the same insertion point (just inside the launch's closing paren). As two
        # separate zero-width edits at one offset they produce "num_stages=2num_warps=8" --
        # found by running the real corpus through the patcher, hence this test.
        out = patch_source(TRAILING_COMMA, {"num_warps": 8, "num_stages": 2})
        ast.parse(out)
        assert "num_warps=8" in out and "num_stages=2" in out
        assert analyze(out).launch_knob_values == {"num_warps": 8, "num_stages": 2}

    def test_mixed_present_and_absent_launch_knobs(self):
        # num_warps is already written; num_stages has to be inserted alongside it.
        out = patch_source(HAS_NUM_WARPS, {"num_warps": 8, "num_stages": 3})
        ast.parse(out)
        assert analyze(out).launch_knob_values == {"num_warps": 8, "num_stages": 3}

    def test_two_knobs_backed_by_one_variable(self):
        # BLOCK and BLOCK_SIZE both read the same assignment, so it must be written once.
        # Writing it twice splices the second value into the first. From the real corpus.
        src = '''
import triton, triton.language as tl

@triton.jit
def k(p, n, BLOCK: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    tl.store(p + tl.arange(0, BLOCK_SIZE), 0.0)

def run(p, n):
    BLOCK = 64
    k[(1,)](p, n, BLOCK=BLOCK, BLOCK_SIZE=BLOCK)
'''
        out = patch_source(src, {"BLOCK": 256, "BLOCK_SIZE": 256, "num_warps": 4})
        ast.parse(out)
        assert "BLOCK = 256" in out
        rep = analyze(out)
        assert rep.knob("BLOCK").current == 256 and rep.knob("BLOCK_SIZE").current == 256

    def test_conflicting_values_on_a_shared_site_is_unpatchable(self):
        src = '''
import triton, triton.language as tl

@triton.jit
def k(p, n, BLOCK: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    tl.store(p + tl.arange(0, BLOCK_SIZE), 0.0)

def run(p, n):
    BLOCK = 64
    k[(1,)](p, n, BLOCK=BLOCK, BLOCK_SIZE=BLOCK)
'''
        # One variable cannot hold two values -- better to skip the config than to emit a
        # kernel whose constants are not what we think they are.
        with pytest.raises(Unpatchable, match="share one site"):
            patch_source(src, {"BLOCK": 64, "BLOCK_SIZE": 128})

    def test_injects_past_a_trailing_comment(self):
        # The comment made a backwards scan think a comma was missing: "BLOCK_SIZE=1, ,".
        out = patch_source(TRAILING_COMMENT, {"BLOCK_SIZE": 64, "num_warps": 1})
        ast.parse(out)
        rep = analyze(out)
        assert rep.knob("BLOCK_SIZE").current == 64
        assert rep.launch_knob_values == {"num_warps": 1}
        assert "# each program processes one element" in out  # comment survives

    def test_tiled_config(self):
        cfg = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
               "num_warps": 8, "num_stages": 4}
        out = patch_source(TILED, cfg)
        rep = analyze(out)
        assert rep.knob("BLOCK_SIZE_M").current == 128
        assert rep.knob("BLOCK_SIZE_N").current == 256
        assert rep.knob("BLOCK_SIZE_K").current == 64
        assert rep.knob("GROUP_SIZE_M").current == 8  # untouched, as designed
        assert "num_stages=4" in out

    def test_unknown_knob_raises(self):
        with pytest.raises(Unpatchable):
            patch_source(ASSIGN_META, {"BLOCK_SIZE_M": 64})

    def test_unparseable_source_raises(self):
        with pytest.raises(Unpatchable):
            patch_source("def broken(:", {"BLOCK_SIZE": 64})


class TestGrids:
    def test_identity_is_first(self):
        assert build_grid(analyze(ASSIGN_META))[0] == {}

    def test_1d_grid_is_blocks_x_warps(self):
        grid = build_grid(analyze(ASSIGN_META))
        assert len(grid) == 25  # identity + 6 blocks x 4 warps
        assert all("num_stages" not in c for c in grid)

    def test_tiled_grid_uses_the_ladder(self):
        grid = build_grid(analyze(TILED))
        assert len(grid) == 15  # identity + 14 rungs
        assert grid[-1]["BLOCK_SIZE_M"] == 256
        assert all("num_stages" in c for c in grid[1:])  # TILED has a loop over K
        assert all("GROUP_SIZE_M" not in c for c in grid[1:])

    def test_untunable_kernel_still_probes_warps(self):
        grid = build_grid(analyze(DYNAMIC_BLOCK))
        assert grid == [{}, {"num_warps": 2}, {"num_warps": 4}, {"num_warps": 8}]

    def test_every_generated_config_applies_cleanly(self):
        for src in (ASSIGN_META, LAUNCH_LITERAL, CLOSURE_GRID, TILED, HAS_NUM_WARPS):
            rep = analyze(src)
            for cfg in build_grid(rep):
                patch_source(src, cfg, rep)  # raises Unpatchable if it does not verify


class TestGamingCheck:
    def test_flags_hardcoded_fed_value(self):
        # The model was told 1024 was best and baked it in, keeping the knob.
        src = ASSIGN_META.replace("BLOCK_SIZE = 128", "BLOCK_SIZE = 1024")
        rep = gaming_report(src, {"BLOCK_SIZE": 1024, "num_warps": 8})
        assert rep["hardcoded_fed_value"] == ["BLOCK_SIZE"]
        assert rep["is_untunable"] is False

    def test_flags_deleted_parameterisation(self):
        # The knob is gone entirely: the tile size is now a magic number in the kernel.
        src = '''
import triton, triton.language as tl

@triton.jit
def k(p, n):
    tl.store(p + tl.arange(0, 1024), 0.0)

def run(p, n):
    k[(1,)](p, n)
'''
        rep = gaming_report(src, {"BLOCK_SIZE": 1024})
        assert rep["is_untunable"] is True
        assert rep["dropped_vs_fed"] == ["BLOCK_SIZE"]
