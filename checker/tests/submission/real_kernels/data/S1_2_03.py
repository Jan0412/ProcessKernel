import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def dw_conv_b32_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_c, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    N, C, H, W, H_out, W_out, K_H, K_W,
    stride_h, stride_w,
    padding_h, padding_w,
    BLOCK_C: tl.constexpr, BLOCK_W: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    ...

def triton_dw_conv_b32(x, weight, bias, stride, padding):
    ...

class ModelNew(nn.Module):
    ...