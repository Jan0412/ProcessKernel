"""F3.2 runtime_loop_bounds."""

from __future__ import annotations

from conftest import src

from ._fixtures import LAUNCH_CONV, RUNTIME_WINDOW, SCALAR_CONV


class TestF32RuntimeLoopBounds:
    def test_fires_when_a_window_bound_is_a_runtime_argument(self, check):
        found = check("F3.2", src(RUNTIME_WINDOW))
        assert len(found) == 1
        assert found[0].data["bounds"] == ["K"]
        assert found[0].severity == "info"
        assert "which arrives as a runtime argument" in found[0].message

    def test_silent_on_the_tiled_reduction_idiom(self, fired):
        """`for k in range(0, K, BLOCK_K)` walks a data extent that must stay dynamic --
        making K constexpr would recompile per shape, so the advice would be wrong."""
        assert not fired(
            "F3.2",
            src(
                '''
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_K: tl.constexpr):
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc += tl.load(a_ptr + k + offs) * tl.load(b_ptr + k + offs)
    tl.store(c_ptr + offs, acc)


class ModelNew(nn.Module):
    def forward(self, a, b):
        c = torch.empty_like(a)
        matmul_kernel[(4,)](a, b, c, 8, 8, 8, BLOCK_K=16)
        return c
'''
            ),
        )

    def test_silent_when_the_bound_is_already_constexpr(self, fired):
        assert not fired("F3.2", src(RUNTIME_WINDOW.replace("K, S, P,", "S, P, K: tl.constexpr,")))

    def test_silent_on_an_unvectorised_kernel(self, fired):
        """No range means F3.1 owns this file; two messages for one fix is noise."""
        assert not fired("F3.2", src(SCALAR_CONV + LAUNCH_CONV))
