"""Family 2 -- memory-traffic / fusion checks.

Not "did it cheat" but "is it slow, and provably so". Everything here is expressed in
bytes and microseconds rather than counts, because a count ("you have too many
kernels") is a suggestion a model can satisfy by fusing the wrong pair, while a byte
figure names a memory transaction that will definitely happen.

The soundness argument for the whole family: **Triton has no cross-launch fusion
pass.** Unlike TorchInductor, nothing merges ``k1[grid](...)`` and ``k2[grid](...)``.
So an intermediate written by one launch and read by the next provably round-trips
through HBM. We are not estimating; we are reading off a guaranteed transaction.

See CHECKS.txt for per-check descriptions and references.
"""

from . import (  # noqa: F401  (import for the side effect of registering)
    f2_1_dead_intermediate,
    f2_2_launch_overhead,
    f2_3_layout_churn,
    f2_4_zeroed_overwritten_buffer,
)
