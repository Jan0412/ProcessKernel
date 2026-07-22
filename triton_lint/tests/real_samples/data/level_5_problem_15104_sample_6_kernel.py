import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pad_same_kernel(
    inp_ptr,               # *float32, input tensor
    out_ptr,               # *float32, output (padded) tensor
    N, C, H, W,            # input dimensions
    out_H, out_W,          # output dimensions (with padding)
    pad_top, pad_left,     # padding offsets
    total_in_elements,     # total number of elements in input
    BLOCK_SIZE: tl.constexpr,
):
    # linear index of the element this program will handle
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offs < total_in_elements

    # load input element
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)

    # decode linear index into N, C, H, W coordinates
    # idx = n * (C*H*W) + c * (H*W) + h * W + w
    c_hw = C * H * W
    h_w = H * W

    n = offs // c_hw
    rem = offs % c_hw
    c = rem // h_w
    rem2 = rem % h_w
    h = rem2 // W
    w = rem2 % W

    # compute output coordinates with padding offset
    out_h = h + pad_top
    out_w = w + pad_left

    # linear offset into output
    out_idx = ((n * C + c) * out_H + out_h) * out_W + out_w

    tl.store(out_ptr + out_idx, x, mask=mask)


def triton_pad_same(x: torch.Tensor, kernel_size, stride):
    """
    Implements the same padding used in PadSameConv2d with a custom Triton kernel.
    Returns a zero‑padded tensor where the original input is copied into the centre.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    x = x.contiguous()

    N, C, H, W = x.shape

    # compute padding amounts (exactly as PadSameConv2d does)
    if isinstance(kernel_size, (tuple, list)):
        k_h, k_w = kernel_size
    else:
        k_h = k_w = kernel_size

    if isinstance(stride, (tuple, list)):
        s_h, s_w = stride
    else:
        s_h = s_w = stride

    pad_y = (s_h * (math.ceil(H / s_h) - 1) + k_h - H) / 2.0
    pad_x = (s_w * (math.ceil(W / s_w) - 1) + k_w - W) / 2.0

    pad_top = int(math.floor(pad_y))
    pad_bottom = int(math.ceil(pad_y))
    pad_left = int(math.floor(pad_x))
    pad_right = int(math.ceil(pad_x))

    out_H = H + pad_top + pad_bottom
    out_W = W + pad_left + pad_right

    # allocate output filled with zeros
    out = torch.zeros((N, C, out_H, out_W), dtype=x.dtype, device=x.device)

    total_in_elements = N * C * H * W
    BLOCK_SIZE = 1024  # a good default, can be tuned

    grid = lambda meta: ((total_in_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    pad_same_kernel[grid](
        x,
        out,
        N,
        C,
        H,
        W,
        out_H,
        out_W,
        pad_top,
        pad_left,
        total_in_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Re‑implementation of PadSameConv2d using a custom Triton kernel for the padding.
    """
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor):
        return triton_pad_same(x, self.kernel_size, self.stride)


# Keep the original helper functions for compatibility with the benchmark harness
def get_inputs():
    # matches the original signature (batch, channels, H, W)
    return [torch.rand([4, 4, 4, 4], device="cuda")]


def get_init_inputs():
    # kernel_size = 4 as in the original example, stride defaults to 1
    return [[], {"kernel_size": 4}]