import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------
# Triton kernel: batched matrix multiplication + bias (FP32)
# --------------------------------------------------------------
@triton.jit
def matmul_bias_kernel(
    a_ptr,          # X: [S, D]
    b_ptr,          # W: [D, E]
    c_ptr,          # output: [S, E]
    bias_ptr,       # bias: [E]
    S, D, E,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)  # block row
    pid_n = tl.program_id(1)  # block col

    # compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # mask for out‑of‑bounds rows / cols
    mask_m = offs_m < S
    mask_n = offs_n < E

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, D, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A and B tiles
        a = tl.load(a_ptr + (offs_m[:, None] * D + offs_k[None, :]),
                    mask=mask_m[:, None] & (offs_k[None, :] < D),
                    other=0.0)
        b = tl.load(b_ptr + (offs_k[:, None] * E + offs_n[None, :]),
                    mask=(offs_k[:, None] < D) & mask_n[None, :],
                    other=0.0)

        # Compute block matrix multiplication
        acc += tl.dot(a, b)

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Write back result
    tl.store(c_ptr + (offs_m[:, None] * E + offs_n[None, :]),
             acc,
             mask=mask_m[:, None] & mask_n[None, :])


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Custom linear layer using Triton for FP32.
    x:    [B, S, D]  (contiguous)
    weight: [D, E]
    bias:   [E]
    Returns y: [B, S, E]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    B, S, D = x.shape
    E = weight.shape[1]

    out = torch.empty((B, S, E), device=x.device, dtype=torch.float32)

    # Tune block sizes (these work well for many sizes)
    BLOCK_M = 64   # rows processed per program
    BLOCK_N = 64   # cols processed per program
    BLOCK_K = 32   # reduction dimension per iteration

    grid_m = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]),)
    grid_n = lambda meta: (triton.cdiv(E, meta["BLOCK_N"]),)

    for b in range(B):
        a_ptr = x[b].contiguous()
        c_ptr = out[b]
        matmul_bias_kernel[grid_m, grid_n](
            a_ptr,
            weight,
            c_ptr,
            bias,
            S, D, E,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    return out


# --------------------------------------------------------------
# Optimized model using the custom Triton kernel
# --------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, dim: int, embed_dim: int):
        super().__init__()
        # keep a regular Linear for parameter storage (weights & bias)
        self.proj = nn.Linear(dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]  (float32, CUDA)
        The original forward does:
            x = x.flatten(2).transpose(1, 2)   # -> [B, H*W, C]
            x = self.proj(x)                   # linear on last dim
        """
        B, C, H, W = x.shape
        # reshape to (B, S, C) where S = H*W
        x = x.permute(0, 2, 3, 1).contiguous()          # [B, H, W, C]
        x = x.view(B, H * W, C)                         # [B, S, C]

        # Use custom Triton linear
        out = triton_linear(x, self.proj.weight, self.proj.bias)
        return out