import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------------

@triton.jit
def sum_kernel(
    x_ptr,                # input tensor (N*C*H*W)
    sum_ptr,              # per‑group sum (N*G)
    N, C, H, W,           # tensor shape
    G, GROUP_SIZE,        # group settings
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = N * C * H * W
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ---- compute indices (NCHW layout, contiguous) ----
    stride_n = C * H * W
    stride_c = H * W
    # n = offsets // stride_n
    n = offsets // stride_n
    residual = offsets % stride_n
    c = residual // stride_c
    # group index
    g = c // GROUP_SIZE

    # linear index into sum tensor
    sum_idx = n * G + g

    # atomic accumulation
    tl.atomic_add(sum_ptr + sum_idx, x, mask=mask)


@triton.jit
def sumsq_kernel(
    x_ptr,
    sumsq_ptr,
    N, C, H, W,
    G, GROUP_SIZE,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = N * C * H * W
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    stride_n = C * H * W
    stride_c = H * W
    n = offsets // stride_n
    residual = offsets % stride_n
    c = residual // stride_c
    g = c // GROUP_SIZE

    sumsq_idx = n * G + g
    tl.atomic_add(sumsq_ptr + sumsq_idx, x * x, mask=mask)


@triton.jit
def norm_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,          # (N*G)
    inv_std_ptr,       # (N*G)
    weight_ptr,        # (C)
    bias_ptr,          # (C)
    N, C, H, W,
    G, GROUP_SIZE,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = N * C * H * W
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    stride_n = C * H * W
    stride_c = H * W
    stride_h = W

    n = offsets // stride_n
    residual = offsets % stride_n
    c = residual // stride_c
    residual2 = residual % stride_c
    # h and w are not needed for the formula, but we compute them for completeness
    # h = residual2 // stride_h
    # w = residual2 % stride_h

    g = c // GROUP_SIZE
    mean = tl.load(mean_ptr + n * G + g, mask=mask, other=0.0)
    inv_std = tl.load(inv_std_ptr + n * G + g, mask=mask, other=0.0)

    weight = tl.load(weight_ptr + c, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)

    # (x - mean) * inv_std * weight + bias
    out = (x - mean) * inv_std
    out = out * weight + bias

    tl.store(out_ptr + offsets, out, mask=mask)


# ------------------------------------------------------------------
# Helper wrappers for launching the kernels
# ------------------------------------------------------------------

def launch_sum(x, sum_tensor, G):
    N, C, H, W = x.shape
    GROUP_SIZE = C // G
    BLOCK_SIZE = 128
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    sum_kernel[grid](
        x, sum_tensor,
        N, C, H, W,
        G, GROUP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def launch_sumsq(x, sumsq_tensor, G):
    N, C, H, W = x.shape
    GROUP_SIZE = C // G
    BLOCK_SIZE = 128
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    sumsq_kernel[grid](
        x, sumsq_tensor,
        N, C, H, W,
        G, GROUP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def launch_norm(x, out, mean, inv_std, weight, bias, G):
    N, C, H, W = x.shape
    GROUP_SIZE = C // G
    BLOCK_SIZE = 128
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    norm_kernel[grid](
        x, out,
        mean, inv_std,
        weight, bias,
        N, C, H, W,
        G, GROUP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ------------------------------------------------------------------
# Optimized Model (ModelNew)
# ------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    GroupNorm implementation where the per‑group mean/variance
    and the final normalization are performed with custom Triton kernels.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous layout on CUDA
        if not x.is_cuda:
            raise RuntimeError("ModelNew only supports CUDA tensors.")
        x = x.contiguous()

        N, C, H, W = x.shape
        G = self.num_groups
        group_size = C // G
        device = x.device
        dtype = x.dtype

        # Allocate buffers for per‑group sums
        sum_tensor = torch.zeros(N, G, device=device, dtype=dtype)
        sumsq_tensor = torch.zeros(N, G, device=device, dtype=dtype)

        # Compute per‑group sum and sum‑of‑squares
        launch_sum(x, sum_tensor, G)
        launch_sumsq(x, sumsq_tensor, G)

        # Derive mean, variance, and inverse standard deviation
        count = group_size * H * W
        mean = sum_tensor / count
        var = sumsq_tensor / count - mean * mean
        inv_std = 1.0 / torch.sqrt(var + self.eps)

        # Prepare weight and bias (fallback to 1/0 if not affine)
        if self.weight is not None:
            weight = self.weight
        else:
            weight = torch.ones(C, device=device, dtype=dtype)
        if self.bias is not None:
            bias = self.bias
        else:
            bias = torch.zeros(C, device=device, dtype=dtype)

        # Output tensor
        out = torch.empty_like(x)

        # Final fused normalization kernel
        launch_norm(x, out, mean, inv_std, weight, bias, G)

        return out