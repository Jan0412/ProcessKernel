"""Family 1 -- fallback / fake-work checks.

Does the solution actually do the work in Triton, or does it hand the work back
to PyTorch (or never run its kernel at all)?  See CHECKS.txt for the description
and literature reference of each check.
"""

from . import (  # noqa: F401  (import for the side effect of registering)
    f1_1_no_triton_kernel,
    f1_2_dead_kernel,
    f1_3_discarded_output,
    f1_4_torch_fallback,
    f1_5_nn_module_call,
    f1_6_passthrough_kernel,
    f1_7_compile_offload,
)
