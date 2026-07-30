import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------------------ Triton kernels ---------------------------------

@triton.jit
def leaky_relu_kernel(
    src_ptr,               # input pointer
    dst_ptr,               # output pointer
    slope,                 # leaky slope (float)
    n_elements,            # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(src_ptr + offsets, mask=mask)
    y = tl.where(x >= 0, x, x * slope)
    tl.store(dst_ptr + offsets, y, mask=mask)


def triton_leaky_relu(x: torch.Tensor, slope: float) -> torch.Tensor:
    """Element‑wise LeakyReLU implemented in Triton."""
    assert x.is_cuda, "triton_leaky_relu works on CUDA tensors only"
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024  # good default for FP32

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    leaky_relu_kernel[grid](
        x,
        out,
        slope,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def cat_kernel(
    src_ptr,               # source tensor pointer
    dst_ptr,               # destination tensor pointer
    dst_offset,            # offset inside destination (in elements)
    n_elements,            # number of elements to copy
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + dst_offset + offsets, val, mask=mask)


def triton_cat(tensors, dim: int = 1) -> torch.Tensor:
    """
    Concatenates tensors along channel dimension (dim=1) using a Triton kernel.
    All tensors must have the same N, H, W and be on the same CUDA device.
    """
    assert dim == 1, "Only channel‑wise concatenation (dim=1) is supported"
    device = tensors[0].device
    dtype = tensors[0].dtype

    N, _, H, W = tensors[0].shape
    total_C = sum(t.shape[1] for t in tensors)

    out = torch.empty((N, total_C, H, W), device=device, dtype=dtype)
    out_flat = out.view(-1)
    dst_ptr = out_flat.data_ptr()

    BLOCK_SIZE = 1024
    offset = 0
    for t in tensors:
        t_cont = t.contiguous()
        src_ptr = t_cont.data_ptr()
        n_elements = t_cont.numel()
        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        cat_kernel[grid](
            src_ptr,
            dst_ptr,
            offset,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        offset += n_elements

    return out


# ------------------------------ Optimized model --------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, internal_size=256,
                 kernel_size=3, leaky_slope=0.02):
        super().__init__()
        pad = kernel_size // 2
        self.leaky_slope = leaky_slope

        self.conv1 = nn.Conv2d(in_channels, internal_size,
                               kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(in_channels + internal_size, internal_size,
                               kernel_size=kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(in_channels + 2 * internal_size,
                               out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # conv1 + leaky ReLU (Triton activation)
        c1 = self.conv1(x)
        x1 = triton_leaky_relu(c1, self.leaky_slope)

        # concatenate x and x1 (Triton cat) -> conv2 + leaky ReLU
        cat12 = triton_cat([x, x1], dim=1)
        c2 = self.conv2(cat12)
        x2 = triton_leaky_relu(c2, self.leaky_slope)

        # final concatenation and conv3 (no activation)
        cat123 = triton_cat([x, x1, x2], dim=1)
        out = self.conv3(cat123)
        return out


# ------------------------------ Helper functions --------------------------------

def get_inputs():
    return [torch.rand([4, 4, 4, 4]).cuda()]


def get_init_inputs():
    # matches the original signature: (in_channels, out_channels)
    return [4, 4]


# The model class expected by the benchmark harness
Model = ModelNew