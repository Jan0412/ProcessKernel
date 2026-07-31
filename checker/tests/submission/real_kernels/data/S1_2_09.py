import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _inst_norm_stats_kernel(
    x_ptr, mean_ptr, var_ptr,
    n, c, d, h, w,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_mean_n, stride_mean_c,
    stride_var_n, stride_var_c,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # ...
    pass

@triton.jit
def _fused_conv_mul_norm_clamp_mul_max_kernel(
    x_ptr, w_ptr, b_ptr, mul_ptr, mean_ptr, var_ptr,
    out_ptr,
    n, c_in, c_out, d_in, h_in, w_in, d_out, h_out, w_out,
    kd, kh, kw,
    pad_d, pad_h, pad_w,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_bn, stride_bc,
    stride_mul_n, stride_mul_c, stride_mul_d, stride_mul_h, stride_mul_w,
    stride_mean_n, stride_mean_c,
    stride_var_n, stride_var_c,
    stride_out_n, stride_out_d, stride_out_h, stride_out_w,
    clamp_min, clamp_max,
    eps,
    BLOCK_N: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_DO: tl.constexpr, BLOCK_HO: tl.constexpr, BLOCK_WO: tl.constexpr
):
    # ...
    pass

def get_inst_norm_stats(x, eps=1e-5):
    # ...
    pass

def fused_operation(x, w, b, multiplier, clamp_min, clamp_max, eps=1e-5):
    # ...
    pass

class ModelNew(nn.Module):
    # ...
    pass