"""Family 3 -- intra-kernel structure.

F1 asks whether the work is real, F2 what the host wastes *between* kernels. F3 asks
whether the kernel body and its launch configuration use the GPU at all.

The soundness argument that carries F2 -- Triton has no cross-launch fusion pass, so the
round trip provably happens -- does not transfer here, so this family is deliberately
narrow. Every check keys on a structural fact of the AST that holds for any shape and any
hardware, never on a tile size or an occupancy target, both of which depend on inputs the
linter cannot see.

See CHECKS.txt for per-check descriptions and references.
"""

from . import (  # noqa: F401  (import for the side effect of registering)
    f3_1_unvectorised_kernel,
    f3_2_runtime_loop_bounds,
)
