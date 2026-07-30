import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------------------
# Triton kernel: compute per‑sample scaling factor
# ------------------------------------------------------------------------
@triton.jit
def compute_scale_kernel(
    x_ptr,          # pointer to input tensor (B, N)
    scale_ptr,      # pointer to output scale (B,)
    max_norm,       # scalar max_norm (float32)
    B, N,           # dimensions
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)                     # one program per batch element
    # pointer to the start of this batch's data
    x_batch_ptr = x_ptr + batch_id * N

    # accumulate sum of squares (scalar per thread, BLOCK_SIZE == 1)
    sum_sq = tl.float32(0.0)
    for offs in range(0, N, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)       # idx within this batch
        mask = idx < N
        x = tl.load(x_batch_ptr + idx, mask=mask, other=0.0)
        sum_sq += tl.where(mask, x * x, 0.0)

    # reduction across the (single) thread – sum_sq already holds the total
    norm = tl.sqrt(sum_sq)                          # L2 norm
    scale = max_norm / norm
    scale = tl.minimum(scale, 1.0)                  # clamp to 1.0
    tl.store(scale_ptr + batch_id, scale)


# ------------------------------------------------------------------------
# Triton kernel: apply the scaling factors
# ------------------------------------------------------------------------
@triton.jit
def apply_scale_kernel(
    x_ptr,          # input tensor (B, N)
    out_ptr,        # output tensor (B, N)
    scale_ptr,      # per‑batch scaling factors (B,)
    B, N,           # dimensions
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < B * N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # derive batch index for each element
    batch_id = offs // N
    scale = tl.load(scale_ptr + batch_id, mask=mask, other=1.0)

    out = x * scale
    tl.store(out_ptr + offs, out, mask=mask)


# ------------------------------------------------------------------------
# Helper wrappers around the kernels
# ------------------------------------------------------------------------
def triton_compute_scale(x: torch.Tensor, max_norm: float):
    """
    Compute per‑sample scaling factors for tensor x of shape (B, C, H, W).
    Returns a tensor of shape (B,).
    """
    assert x.is_cuda and x.dtype == torch.float32
    B, C, H, W = x.shape
    N = C * H * W

    scale = torch.empty(B, device=x.device, dtype=torch.float32)

    # one program per batch element, BLOCK_SIZE = 1 (scalar reduction)
    grid = lambda meta: (B,)
    compute_scale_kernel[grid](
        x,
        scale,
        max_norm,
        B,
        N,
        BLOCK_SIZE=1,
    )
    return scale


def triton_apply_scale(x: torch.Tensor, scale: torch.Tensor):
    """
    Multiply each sample in x by its corresponding scaling factor.
    x: (B, C, H, W)
    scale: (B,)
    Returns a new tensor with the same shape as x.
    """
    assert x.is_cuda and x.dtype == torch.float32
    B, C, H, W = x.shape
    N = C * H * W

    out = torch.empty_like(x)

    BLOCK_SIZE = 128  # tunable
    total_elements = B * N
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    apply_scale_kernel[grid](
        x,
        out,
        scale,
        B,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ------------------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, max_norm: float):
        super().__init__()
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor):
        # ensure data is on CUDA and contiguous
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        # compute per‑sample scaling factors using Triton
        scale = triton_compute_scale(x, self.max_norm)
        # apply scaling
        out = triton_apply_scale(x, scale)
        return out


# Preserve the original name expected by the KernelBench adapter
Model = ModelNew