import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    X, W, B, Y,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # ... standard tiling ...
    pass

@triton.jit
def bn_act_kernel(
    X, Y,
    running_mean, running_var, weight, bias,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # ... flat indexing ...
    pass

class ModelNew(nn.Module):
    ...