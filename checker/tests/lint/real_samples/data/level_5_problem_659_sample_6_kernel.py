import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------

@triton.jit
def batched_matmul_kernel(
    A,                     # [B, M, K]
    B,                     # [B, K, N]
    C,                     # [B, M, N]
    BATCH, M, N, K,
    stride_aa, stride_ab, stride_ac,   # strides for A
    stride_ba, stride_bb, stride_bc,   # strides for B
    stride_ca, stride_cb,              # strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched matrix multiplication: C[b] = A[b] @ B[b]"""
    pid = tl.program_id(0)

    # ------------------------------
    # Compute batch, block row, block col indices
    # ------------------------------
    batch_id = pid // ((M + BLOCK_M - 1) // BLOCK_M * (N + BLOCK_N - 1) // BLOCK_N)
    block_id = pid % ((M + BLOCK_M - 1) // BLOCK_M * (N + BLOCK_N - 1) // BLOCK_N)
    pid_m = block_id // ((N + BLOCK_N - 1) // BLOCK_N)
    pid_n = block_id % ((N + BLOCK_N - 1) // BLOCK_N)

    # ------------------------------
    # Offsets for the block we're computing
    # ------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Mask to avoid out‑of‑bounds accesses
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # Load A and B tiles
    a_ptrs = (A + batch_id * stride_aa
                + offs_m[:, None] * stride_ab
                + offs_k[None, :] * stride_ac)
    b_ptrs = (B + batch_id * stride_ba
                + offs_k[:, None] * stride_bb
                + offs_n[None, :] * stride_bc)

    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

    # Compute the matrix multiplication for the tile
    acc = tl.dot(a, b)

    # Write back the result
    c_ptrs = (C + batch_id * stride_ca
                + offs_m[:, None] * stride_cb
                + offs_n[None, :] * stride_cb)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication using Triton.
    A : (B, M, K)
    B : (B, K, N)
    Returns C : (B, M, N)
    """
    assert A.is_cuda and B.is_cuda
    BATCH, M, K = A.shape
    _, K2, N = B.shape
    assert K == K2

    # Prepare output
    C = torch.empty((BATCH, M, N), dtype=A.dtype, device=A.device)

    # Block sizes (tuned for typical sizes; can be adjusted)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Compute strides (in elements, not bytes)
    stride_aa, stride_ab, stride_ac = A.stride()
    stride_ba, stride_bb, stride_bc = B.stride()
    stride_ca, stride_cb, _ = C.stride()

    # Grid: one program per (batch, block_m, block_n)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    total_progs = BATCH * grid_m * grid_n
    grid = (total_progs,)

    batched_matmul_kernel[grid](
        A, B, C,
        BATCH, M, N, K,
        stride_aa, stride_ab, stride_ac,
        stride_ba, stride_bb, stride_bc,
        stride_ca, stride_cb,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


@triton.jit
def scale_kernel(
    x_ptr,                # input pointer
    out_ptr,              # output pointer
    scale,                # scalar (float)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x / scale  (element‑wise)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x / scale
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Divide tensor x by a scalar using Triton."""
    assert x.is_cuda
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    scale_kernel[grid](x, out, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


# --------------------------------------------------------------
# Optimized Model
# --------------------------------------------------------------

class ScaledDotProductAttentionNew(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v : (B, S, D)
        mask    : (B, S, S) with True where the position should be masked
        Returns (output, attn)
        """
        # ---- 1. QK^T -------------------------------------------------
        # Triton batched matmul
        scores = triton_bmm(q, k.transpose(1, 2))          # (B, S, S)

        # ---- 2. Scaling ------------------------------------------------
        if self.scale:
            scale_factor = math.sqrt(k.shape[-1])
            scores = triton_scale(scores, scale_factor)   # (B, S, S)

        # ---- 3. Mask ---------------------------------------------------
        if mask is not None:
            # mask is bool; we use a large negative value for masked positions
            scores = scores.masked_fill(mask, -1e9)

        # ---- 4. Softmax ------------------------------------------------
        attn = self.softmax(scores)                       # (B, S, S)

        # ---- 5. Dropout (optional) ------------------------------------
        if self.dropout is not None:
            attn = self.dropout(attn)

        # ---- 6. Attn @ V -----------------------------------------------
        output = triton_bmm(attn, v)                      # (B, S, D)

        return output, attn


# --------------------------------------------------------------
# Compatibility alias (as required by the original benchmark harness)
# --------------------------------------------------------------

ModelNew = ScaledDotProductAttentionNew