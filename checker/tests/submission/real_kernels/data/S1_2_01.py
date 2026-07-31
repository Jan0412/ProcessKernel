import torch
import torch.nn as nn
import triton
import triton.language as tl

# Kernel
@triton.jit
def mul_scalar_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scalar,
    BLOCK_SIZE: tl.constexpr,
):
    ...

# Wrapper
def triton_mul_scalar(input: torch.Tensor, scalar: float) -> torch.Tensor:
    ...

# Model
class ModelNew(nn.Module):
    ...

# Inputs helpers
M = ...
N = ...
def get_inputs():
    ...
def get_init_inputs():
    ...