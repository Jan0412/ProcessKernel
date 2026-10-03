import torch
import torch.nn as nn
import triton
import triton.language as tl

# Kernel
@triton.jit
def argmax_kernel(x_ptr, out_ptr, M, BLOCK_M: tl.constexpr):
    # ...
    pass

# Wrapper
def triton_argmax(x, dim):
    # ...
    pass

# Model Classes
class Model(nn.Module):
    ...

class ModelNew(nn.Module):
    ...

# Input generators
def get_inputs(): ...
def get_init_inputs(): ...