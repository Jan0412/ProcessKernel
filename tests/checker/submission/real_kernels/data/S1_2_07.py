import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel for fused GEMM + activation
@triton.jit
def fused_conv_activation_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    ...

# Wrapper function
def fused_conv_relu_hardswish(x, weight, bias):
    ...

# Model class
class ModelNew(nn.Module):
    ...