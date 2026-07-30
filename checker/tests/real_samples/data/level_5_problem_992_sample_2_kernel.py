import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------

@triton.jit
def copy_kernel(
    src_ptr,          # pointer to source tensor
    dst_ptr,          # pointer to destination tensor
    src_offset,       # offset in source (in elements)
    n_elements,       # number of elements to copy
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load from source with the given offset
    src = tl.load(src_ptr + src_offset + offsets, mask=mask, other=0.0)
    # store to destination
    tl.store(dst_ptr + offsets, src, mask=mask)


# ------------------------------------------------------------
# Python wrappers around the kernels
# ------------------------------------------------------------

def triton_split(x: torch.Tensor):
    """
    Splits a tensor of shape (N, C, H, W) into two tensors along the channel dimension.
    Assumes C is even.
    """
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C, H, W = x.shape
    assert C % 2 == 0, "Channel dimension must be even for this split."
    half = C // 2

    # output tensors
    out1 = torch.empty((N, half, H, W), dtype=x.dtype, device=x.device)
    out2 = torch.empty((N, half, H, W), dtype=x.dtype, device=x.device)

    total_half = N * half * H * W
    BLOCK_SIZE = 1024

    grid = lambda meta: ((total_half + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # copy first half
    copy_kernel[grid](
        x,
        out1,
        0,                      # src_offset
        total_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # copy second half (starts after the first half in the flattened view)
    copy_kernel[grid](
        x,
        out2,
        total_half,             # src_offset
        total_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out1, out2


def triton_cat(x1: torch.Tensor, x2: torch.Tensor):
    """
    Concatenates two tensors of shape (N, C/2, H, W) along the channel dimension,
    reproducing torch.cat((x1, x2), dim=1).
    """
    assert x1.is_cuda and x2.is_cuda
    assert x1.dtype == torch.float32 and x2.dtype == torch.float32
    x1 = x1.contiguous()
    x2 = x2.contiguous()

    N, C_half, H, W = x1.shape
    assert x2.shape == (N, C_half, H, W), "Input shapes must match for concatenation."

    out = torch.empty((N, C_half * 2, H, W), dtype=x1.dtype, device=x1.device)

    total1 = x1.numel()
    BLOCK_SIZE = 1024

    grid1 = lambda meta: ((total1 + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # copy first tensor
    copy_kernel[grid1](
        x1,
        out,
        0,               # src_offset
        total1,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # copy second tensor (starts after first tensor in destination)
    copy_kernel[grid1](
        x2,
        out,
        total1,          # src_offset in destination is total1
        total1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Replaces the original Python split with a Triton‑accelerated version.
        """
        return triton_split(x)

    def inverse(self, x1, x2):
        """
        Replaces torch.cat with a Triton‑accelerated concatenation.
        """
        return triton_cat(x1, x2)


# ------------------------------------------------------------
# Compatibility with the original interface
# ------------------------------------------------------------
# The original script expected a variable called `Model` that points to the class.
Model = ModelNew