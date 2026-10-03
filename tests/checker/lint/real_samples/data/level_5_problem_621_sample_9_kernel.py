import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -------------------------------------------------
# Triton kernel for batched matrix multiplication
# -------------------------------------------------
@triton.jit
def bmm_kernel(
    A_ptr,                # Pointer to A [B, M, K]
    B_ptr,                # Pointer to B [B, K, N]
    C_ptr,                # Pointer to output C [B, M, N]
    stride_a_batch,       # stride for batch dimension in A
    stride_a_m,           # stride for M dimension in A
    stride_a_k,           # stride for K dimension in A
    stride_b_batch,       # stride for batch dimension in B
    stride_b_k,           # stride for K dimension in B
    stride_b_n,           # stride for N dimension in B
    stride_c_batch,       # stride for batch dimension in C
    stride_c_m,           # stride for M dimension in C
    stride_c_n,           # stride for N dimension in C
    M, N, K,              # matrix sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers to the start of the batch
    a_batch_ptr = A_ptr + pid_batch * stride_a_batch
    b_batch_ptr = B_ptr + pid_batch * stride_b_batch
    c_batch_ptr = C_ptr + pid_batch * stride_c_batch

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A tile [BLOCK_M, BLOCK_K]
        a = tl.load(
            a_batch_ptr
            + (offs_m[:, None] * stride_a_m)
            + (offs_k[None, :] * stride_a_k),
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        # Load B tile [BLOCK_K, BLOCK_N]
        b = tl.load(
            b_batch_ptr
            + (offs_k[:, None] * stride_b_k)
            + (offs_n[None, :] * stride_b_n),
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        # Compute block multiplication
        acc += tl.dot(a, b)

    # Store the result
    tl.store(
        c_batch_ptr
        + (offs_m[:, None] * stride_c_m)
        + (offs_n[None, :] * stride_c_n),
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def triton_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication using Triton.
    a: [B, M, K]
    b: [B, K, N]
    returns: [B, M, N]
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2, "Incompatible shapes for batched matmul."

    out = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    # Compute strides (in elements, not bytes)
    stride_a_batch = a.stride(0)
    stride_a_m = a.stride(1)
    stride_a_k = a.stride(2)

    stride_b_batch = b.stride(0)
    stride_b_k = b.stride(1)
    stride_b_n = b.stride(2)

    stride_c_batch = out.stride(0)
    stride_c_m = out.stride(1)
    stride_c_n = out.stride(2)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        B,
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    bmm_kernel[grid](
        a,
        b,
        out,
        stride_a_batch,
        stride_a_m,
        stride_a_k,
        stride_b_batch,
        stride_b_k,
        stride_b_n,
        stride_c_batch,
        stride_c_m,
        stride_c_n,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


# -------------------------------------------------
# Optimized model using the Triton batched matmul
# -------------------------------------------------
class ModelNew(nn.Module):
    """Scaled Dot Production with Triton‑accelerated batched matmul"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value):
        """
        query: [batch, d_k, d_out]
        key:   [batch, d_k, n_candidate]
        value: [batch, d_v, n_candidate]
        """
        # query^T : [batch, d_out, d_k]
        query_t = query.transpose(2, 1)
        # attn = softmax( (Q^T K) / temperature )
        attn = triton_bmm(query_t, key)                # [B, d_out, n_candidate]
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # value^T : [batch, n_candidate, d_v]
        value_t = value.transpose(2, 1)
        # output = attn * V^T
        output = triton_bmm(attn, value_t)             # [B, d_out, d_v]

        return output, attn