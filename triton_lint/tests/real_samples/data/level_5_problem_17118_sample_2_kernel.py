import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------- Triton kernels ------------------- #

@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    """In‑place ReLU implemented with Triton."""
    assert x.is_cuda, "Triton kernels require CUDA tensors"
    x = x.contiguous()
    out = torch.empty_like(x)

    n_el = x.numel()
    BLOCK = 128
    grid = lambda meta: ((n_el + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    relu_kernel[grid](x, out, n_el, BLOCK_SIZE=BLOCK)
    return out


@triton.jit
def max_pool2d_kernel(
    inp_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    H_out,
    W_out,
    stride_h,
    stride_w,
    kernel_h,
    kernel_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C * H_out * W_out)

    # decode linear offset into (n, c, h_out, w_out)
    n = offsets // (C * H_out * W_out)
    rem = offsets % (C * H_out * W_out)
    c = rem // (H_out * W_out)
    rem = rem % (H_out * W_out)
    h_out = rem // W_out
    w_out = rem % W_out

    # compute input window start indices
    h_start = h_out * stride_h
    w_start = w_out * stride_w

    max_val = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)

    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_in = h_start + kh
            w_in = w_start + kw
            # bounds check
            in_bounds = (h_in < H) & (w_in < W)
            # compute flat index into input
            inp_index = (
                n * (C * H * W)
                + c * (H * W)
                + h_in * W
                + w_in
            )
            val = tl.load(inp_ptr + inp_index, mask=in_bounds & mask, other=float("-inf"))
            max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_max_pool2d(x: torch.Tensor, kernel_size=3, stride=2) -> torch.Tensor:
    """2‑D max‑pool with square kernel and stride, implemented in Triton."""
    assert x.is_cuda, "Triton kernels require CUDA tensors"
    N, C, H, W = x.shape
    assert isinstance(kernel_size, int) and isinstance(stride, int)
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    out = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)

    total = N * C * H_out * W_out
    BLOCK = 128
    grid = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    max_pool2d_kernel[grid](
        x,
        out,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        stride,
        stride,
        kernel_size,
        kernel_size,
        BLOCK_SIZE=BLOCK,
    )
    return out


@triton.jit
def copy_kernel(
    src_ptr,
    dst_ptr,
    src_offset,
    dst_offset,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    src = tl.load(src_ptr + src_offset + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offset + offsets, src, mask=mask)


def triton_cat(tensors, dim=1):
    """Concatenate a list of tensors along channel dimension (dim=1) via Triton."""
    assert all(t.is_cuda for t in tensors)
    # assume all tensors have same N, H, W and dim == 1
    N, _, H, W = tensors[0].shape
    total_c = sum(t.shape[1] for t in tensors)
    out = torch.empty((N, total_c, H, W), dtype=tensors[0].dtype, device=tensors[0].device)

    dst_offset = 0
    BLOCK = 128
    for t in tensors:
        n_el = t.numel()
        grid = lambda meta: ((n_el + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        copy_kernel[grid](
            t,
            out,
            0,
            dst_offset,
            n_el,
            BLOCK_SIZE=BLOCK,
        )
        dst_offset += n_el
    return out


# ------------------- Model definition ------------------- #

class Conv2dTriton(nn.Module):
    """Conv2d wrapper that uses Triton for the fused ReLU (and optional BatchNorm)."""

    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = triton_relu(x)          # Triton ReLU
        return x


class GridReduction2Triton(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3_1 = Conv2dTriton(in_channels, 192, 1)
        self.branch3x3_2 = Conv2dTriton(192, 320, 3, stride=2)
        self.branch7x7x3_1 = Conv2dTriton(in_channels, 192, 1)
        self.branch7x7x3_2 = Conv2dTriton(192, 192, (1, 7), padding=(0, 3))
        self.branch7x7x3_3 = Conv2dTriton(192, 192, (7, 1), padding=(3, 0))
        self.branch7x7x3_4 = Conv2dTriton(192, 192, 3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = triton_max_pool2d(x, kernel_size=3, stride=2)

        # Triton based concatenation
        out = triton_cat([branch3x3, branch7x7x3, branch_pool], dim=1)
        return out


# Exported model name as requested
ModelNew = GridReduction2Triton