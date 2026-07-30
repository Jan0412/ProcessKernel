import torch
import torch.nn as nn
import triton
import triton.language as tl


def get_outnorm(x: torch.Tensor, out_norm: str = "") -> torch.Tensor:
    """Common function to get a loss normalization value."""
    img_shape = x.shape
    if not out_norm:
        return 1.0
    norm = 1.0
    if "b" in out_norm:
        norm /= img_shape[0]
    if "c" in out_norm:
        norm /= img_shape[-3]
    if "i" in out_norm:
        norm /= img_shape[-1] * img_shape[-2]
    return norm


@triton.jit
def gram_kernel(
    a_ptr,          # pointer to (c, n) matrix (row‑major)
    out_ptr,        # pointer to (c, c) output matrix
    c,              # number of rows / cols (= channels)
    n,              # inner dimension (= h*w)
    norm,           # normalization scalar (float32)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask for valid output positions
    mask_m = offs_m < c
    mask_n = offs_n < c

    # Accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the reduction dimension
    for k in range(0, n, BLOCK_K):
        cur_k = tl.minimum(BLOCK_K, n - k)  # size of the current k‑tile

        # Offsets for the current k‑tile
        offs_k = k + tl.arange(0, cur_k)

        # Load A tile (shape: BLOCK_M × cur_k)
        a = tl.load(
            a_ptr + (offs_m[:, None] * n + offs_k[None, :]),
            mask=mask_m[:, None] & (offs_k[None, :] < n),
            other=0.0,
        )

        # Load B tile = Aᵀ (shape: cur_k × BLOCK_N)
        b = tl.load(
            a_ptr + (offs_k[:, None] * n + offs_n[None, :]),
            mask=(offs_k[:, None] < n) & mask_n[None, :],
            other=0.0,
        )

        # Matrix‑multiply the tiles and accumulate
        acc += tl.dot(a, b)   # (BLOCK_M, BLOCK_N)

    # Apply normalization
    acc = acc * norm

    # Store the result
    out = acc
    tl.store(
        out_ptr + (offs_m[:, None] * c + offs_n[None, :]),
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def triton_gram(x: torch.Tensor, out_norm: str = "ci") -> torch.Tensor:
    """
    Compute Gram matrix for a batch of feature maps using a fused Triton kernel.
    Input shape: (b, c, h, w)
    Output shape: (b, c, c)
    """
    assert x.is_cuda, "Input must be on CUDA"
    b, c, h, w = x.shape
    n = h * w

    # Normalization factor (scalar)
    norm = float(get_outnorm(x, out_norm))

    # Prepare tensors
    x_flat = x.reshape(b, c, n).contiguous()
    out = torch.empty((b, c, c), device=x.device, dtype=x.dtype)

    # Tunable tile sizes – these work well for typical small‑to‑medium channel counts
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    # Launch one kernel per batch element
    for batch_idx in range(b):
        a_ptr = x_flat[batch_idx].data_ptr()
        out_ptr = out[batch_idx].data_ptr()
        grid = (
            (c + BLOCK_M - 1) // BLOCK_M,
            (c + BLOCK_N - 1) // BLOCK_N,
        )
        gram_kernel[grid](
            a_ptr,
            out_ptr,
            c,
            n,
            norm,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    return out


class ModelNew(nn.Module):
    """
    Optimized Gram matrix computation using a custom Triton kernel.
    """

    def __init__(self, out_norm: str = "ci"):
        super().__init__()
        self.out_norm = out_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (b, c, h, w)
        Returns: Gram matrix of shape (b, c, c) with optional normalization.
        """
        return triton_gram(x, self.out_norm)