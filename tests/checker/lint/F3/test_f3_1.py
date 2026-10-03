"""F3.1 unvectorised_kernel."""

from __future__ import annotations

from conftest import GOOD_KERNEL_FILE, src

from ._fixtures import LAUNCH_CONV, SCALAR_CONV


class TestF31UnvectorisedKernel:
    def test_fires_on_one_element_per_program_kernel(self, check):
        found = check("F3.1", src(SCALAR_CONV + LAUNCH_CONV))
        assert len(found) == 1
        assert found[0].data["kernel"] == "conv_kernel"
        assert "tl.arange" in found[0].message

    def test_silent_on_a_blocked_kernel(self, fired):
        """tl.arange is the whole point: a blocked kernel is what we are asking for."""
        assert not fired("F3.1", GOOD_KERNEL_FILE)

    def test_silent_when_the_kernel_is_never_launched(self, fired):
        """A kernel that never runs is F1.2's finding, not a vectorisation complaint."""
        assert not fired("F3.1", src(SCALAR_CONV))

    def test_silent_on_a_kernel_that_touches_no_memory(self, fired):
        """No load and no store: there is nothing to vectorise, and F1 covers the rest."""
        assert not fired(
            "F3.1",
            src(
                '''
@triton.jit
def noop_kernel(x_ptr, n):
    pid = tl.program_id(0)


class ModelNew(nn.Module):
    def forward(self, x):
        noop_kernel[(4,)](x, x.numel())
        return x
'''
            ),
        )

    def test_a_scalar_loop_over_data_is_worse_than_a_scalar_epilogue(self, check):
        """A loop means the whole tensor is walked one element at a time; an epilogue
        that reads a few precomputed scalars wastes far less, so it must not carry the
        same severity or a model would spend its fix on the wrong kernel."""
        looped = check("F3.1", src(SCALAR_CONV + LAUNCH_CONV))
        epilogue = check(
            "F3.1",
            src(
                '''
@triton.jit
def epilogue_kernel(sum_ptr, gamma_ptr, out_ptr, C):
    pid = tl.program_id(0)
    s = tl.load(sum_ptr + pid)
    g = tl.load(gamma_ptr + pid % C)
    tl.store(out_ptr + pid, s * g)


class ModelNew(nn.Module):
    def forward(self, x, g):
        out = torch.empty_like(x)
        epilogue_kernel[(8,)](x, g, out, 4)
        return out
'''
            ),
        )
        assert [f.severity for f in looped] == ["warn"]
        assert [f.severity for f in epilogue] == ["info"]
