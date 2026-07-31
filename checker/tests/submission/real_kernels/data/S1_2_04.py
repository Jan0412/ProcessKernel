import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose_2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_h, in_w, in_c,
    out_h, out_w, out_c,
    kh, kw,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    ...

def triton_conv_transpose2d(input, weight, bias, stride, padding):
    ...

@triton.jit
def max_pool_2d_kernel(
    input_ptr, output_ptr,
    h, w, c,
    kh, kw,
    stride_h, stride_w,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    ...

def triton_max_pool2d(input, kernel_size, stride):
    ...

class ModelNew(nn.Module):
    ...