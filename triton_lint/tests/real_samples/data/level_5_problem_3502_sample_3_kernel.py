import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------
# Triton kernel: fused Linear (A @ B.T + bias) + Sigmoid
# ------------------------------------------------------------
@triton.jit
def linear_sigmoid_kernel(
    A_ptr,          # [M, K] input matrix
    B_ptr,          # [N, K] weight.T matrix  (transposed weight)
    C_ptr,          # [M, N] output matrix
    bias_ptr,       # [N] bias vector
    M, N, K,
    stride_am, stride_ak,      # strides for A
    stride_bn, stride_bk,      # strides for B (note: B is transposed)
    stride_cm, stride_cn,      # strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the row and column offsets of the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask for out‑of‑bounds rows/cols
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers to the start of the block
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_n[:, None] * stride_bn + tl.arange(0, BLOCK_K)[None, :] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        cur_k = min(BLOCK_K, K - k)

        a = tl.load(a_ptrs, mask=mask_m[:, None] & (tl.arange(0, cur_k)[None, :] < cur_k), other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[:, None] & (tl.arange(0, cur_k)[None, :] < cur_k), other=0.0)

        acc += tl.dot(a, b)

        # Increment pointers to next K‑tile
        a_ptrs += cur_k * stride_ak
        b_ptrs += cur_k * stride_bk

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + offs_n * stride_bn, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Sigmoid activation
    acc = 1.0 / (1.0 + tl.math.exp(-acc))

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_linear_sigmoid(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Linear (x @ weight.T + bias) + Sigmoid using Triton.
    Works for any leading dimensions; only the last dimension must match weight.shape[1].
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Transpose weight to get shape [out_features, in_features] -> [out_features, in_features] (we need weight.T)
    weight_t = weight.t().contiguous()   # shape [out, in]

    # Save original shape to reshape later
    orig_shape = x.shape
    M = x.numel() // weight.shape[1]            # total rows after flattening
    K = weight.shape[1]                         # in_features
    N = weight.shape[0]                         # out_features

    # Flatten input to 2‑D [M, K]
    A = x.view(M, K)

    # Allocate output
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Strides (in elements)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bn = weight_t.stride(0)   # N stride
    stride_bk = weight_t.stride(1)   # K stride
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Tunable block sizes (chosen to fit typical GPUs)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    linear_sigmoid_kernel[grid](
        A,
        weight_t,
        C,
        bias,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Reshape back to original leading dimensions with new last dim N
    return C.view(*orig_shape[:-1], N)


# ------------------------------------------------------------
# Optimized AutoEncoder using the fused Triton kernel
# ------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_question: int, k: int) -> None:
        super().__init__()
        self.g = nn.Linear(num_question, k, bias=True)
        self.h = nn.Linear(k, num_question, bias=True)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2."""
        g_w_norm = torch.norm(self.g.weight, p=2) ** 2
        h_w_norm = torch.norm(self.h.weight, p=2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using fused Linear+Sigmoid kernels.
        """
        # First Linear + Sigmoid
        out = triton_linear_sigmoid(inputs, self.g.weight, self.g.bias)
        # Second Linear + Sigmoid
        out = triton_linear_sigmoid(out, self.h.weight, self.h.bias)
        return out