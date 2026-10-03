import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q, K, V,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Start index for the current batch and head
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    z = off_hz // H

    # Pointers to Q, K, V
    q_offset = (z * stride_qz + off_h * stride_qh + start_m * stride_qm)
    k_offset = (z * stride_kz + off_h * stride_kh)
    v_offset = (z * stride_vz + off_h * stride_vh)
    
    q_ptrs = q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, BLOCK_K)[None, :] * stride_qk
    k_ptrs = k_offset + tl.arange(0, BLOCK_K)[:, None] * stride_kk + tl.arange(0, BLOCK_N)[None, :] * stride_kn
    v_ptrs = v_offset + tl.arange(0, BLOCK_N)[:, None] * stride_vn + tl.arange(0, BLOCK_K)[None, :] * stride_vk

    # Initialize output accumulators
    # m_i keeps track of the max value in the softmax input
    # l_i keeps track of the sum of exponentials
    # acc holds the accumulated output
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Load Q once
    q = tl.load(Q + q_ptrs, mask=(start_m + tl.arange(0, BLOCK_M))[:, None] < M, other=0.0)

    # Scale factor for dot product
    scale = 1.0 / (K ** 0.5)

    # Iterate over K and V in chunks
    for start_n in range(0, N, BLOCK_N):
        # Mask for K and V loading
        mask_n = (start_n + tl.arange(0, BLOCK_N)) < N
        mask_k = (tl.arange(0, BLOCK_K)) < K

        # Load K and V
        k = tl.load(K + k_offset + tl.arange(0, BLOCK_K)[:, None] * stride_kk + (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_kn,
                    mask=(mask_k[:, None]) & (mask_n[None, :]), other=0.0)
        v = tl.load(V + v_offset + (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_vn + tl.arange(0, BLOCK_K)[None, :] * stride_vk,
                    mask=(mask_n[:, None]) & (mask_k[None, :]), other=0.0)

        # Compute Q @ K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= scale

        # Online softmax: update max and sum
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Update accumulated output
        acc = acc * alpha[:, None] + tl.dot(p, v)

        # Update m_i
        m_i = m_i_new

    # Finalize output
    acc = acc / l_i[:, None]

    # Store output
    out_offset = (z * stride_oz + off_h * stride_oh + start_m * stride_om)
    out_ptrs = out_offset + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, BLOCK_N)[None, :] * stride_on
    tl.store(Out + out_ptrs, acc, mask=(start_m + tl.arange(0, BLOCK_M))[:, None] < M)


def scaled_dot_product_attention_triton(Q, K, V):
    """
    Custom Triton kernel for scaled dot-product attention.
    Optimized for FP32 precision.
    """
    # Ensure inputs are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Get dimensions
    batch_size, num_heads, seq_len, embed_dim = Q.shape

    # Output tensor
    Out = torch.empty_like(Q)

    # Block sizes (tunable)
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32

    # Grid configuration
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)

    # Launch kernel
    attention_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        batch_size, num_heads, seq_len, seq_len, embed_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_STAGES=4,
    )

    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Replace PyTorch's scaled_dot_product_attention with our custom Triton kernel
        out = scaled_dot_product_attention_triton(Q, K, V)
        return out