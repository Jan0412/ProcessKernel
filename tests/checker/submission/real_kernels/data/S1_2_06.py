import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def post_process_kernel(
    x_ptr, out_ptr, n_elements,
    add_val, mul_val,
    BLOCK_SIZE: tl.constexpr
): ...

def fused_post_process(x, add_val, mul_val): ...

class ModelNew(nn.Module):
   ...