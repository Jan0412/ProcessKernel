import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# --------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------

@triton.jit
def matmul_bias_relu_kernel(
    A_ptr,          # [M, K]
    B_ptr,          # [K, N] (weight transposed)
    C_ptr,          # [M, N] output
    bias_ptr,       # [N]
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A and B tiles
        A = tl.load(
            A_ptr + (offs_m[:, None] * K + offs_k[None, :]),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        B = tl.load(
            B_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        a += tl.dot(A, B)

    # add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    a += bias[None, :]

    # ReLU
    a = tl.maximum(a, 0.0)

    # Store
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask = mask_m & mask_n
    tl.store(C_ptr + (offs_m[:, None] * N + offs_n[None, :]), a, mask=mask)


@triton.jit
def matmul_bias_kernel(
    A_ptr,          # [M, K]
    B_ptr,          # [K, N] (weight transposed)
    C_ptr,          # [M, N] output
    bias_ptr,       # [N]
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        A = tl.load(
            A_ptr + (offs_m[:, None] * K + offs_k[None, :]),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        B = tl.load(
            B_ptr + (offs_k[:, None] * N + offs_n[None, :]),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        a += tl.dot(A, B)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    a += bias[None, :]

    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask = mask_m & mask_n
    tl.store(C_ptr + (offs_m[:, None] * N + offs_n[None, :]), a, mask=mask)


@triton.jit
def combine_kernel(
    a_ptr,          # [M, N]
    v_ptr,          # [M, N]
    out_ptr,        # [M, N]
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    a = tl.load(a_ptr + (offs_m[:, None] * N + offs_n[None, :]), mask=mask, other=0.0)
    v = tl.load(v_ptr + (offs_m[:, None] * N + offs_n[None, :]), mask=mask, other=0.0)

    # row-wise mean of a
    row_sum = tl.sum(a, axis=1)          # [BLOCK_M]
    row_mean = row_sum / N                # broadcast scalar per row
    row_mean = row_mean[:, None]          # [BLOCK_M, 1]

    out = a + v - row_mean

    tl.store(out_ptr + (offs_m[:, None] * N + offs_n[None, :]), out, mask=mask)


# --------------------------------------------------------------
# Wrapper functions
# --------------------------------------------------------------

def triton_linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    x: (M, K)
    weight: (out_features, in_features) -> we use weight.T of shape (K, N)
    bias: (out_features,)
    Returns: (M, N) = relu(x @ weight.T + bias)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight_t = weight.t().contiguous()   # (K, N)
    bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    matmul_bias_relu_kernel[grid](
        x,
        weight_t,
        out,
        bias,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Linear without activation: out = x @ weight.T + bias
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight_t = weight.t().contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    matmul_bias_kernel[grid](
        x,
        weight_t,
        out,
        bias,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


def triton_combine(a: torch.Tensor, v: torch.Tensor):
    """
    Implements: out = a + v - mean_a_row (broadcasted)
    a, v shape: (M, N)
    """
    assert a.is_cuda and v.is_cuda
    a = a.contiguous()
    v = v.contiguous()

    M, N = a.shape
    out = torch.empty_like(a)

    BLOCK_M = 128
    BLOCK_N = 128

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    combine_kernel[grid](
        a,
        v,
        out,
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out


# --------------------------------------------------------------
# Optimized model
# --------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, STATE_NUM, ACTION_NUM):
        super(ModelNew, self).__init__()
        self.ACTION_NUM = ACTION_NUM
        # keep original Linear modules to reuse parameters & initialization
        self.fc1_a = nn.Linear(in_features=STATE_NUM, out_features=512, bias=True)
        self.fc1_v = nn.Linear(in_features=STATE_NUM, out_features=512, bias=True)
        self.fc2_a = nn.Linear(in_features=512, out_features=ACTION_NUM, bias=True)
        self.fc2_v = nn.Linear(in_features=512, out_features=1, bias=True)

    def forward(self, x):
        # x : (batch, STATE_NUM)
        a = triton_linear_relu(x, self.fc1_a.weight, self.fc1_a.bias)   # (B, 512)
        v = triton_linear_relu(x, self.fc1_v.weight, self.fc1_v.bias)   # (B, 512)

        a = triton_linear(a, self.fc2_a.weight, self.fc2_a.bias)       # (B, ACTION_NUM)
        v = triton_linear(v, self.fc2_v.weight, self.fc2_v.bias)       # (B, 1)

        # expand v to match action dimension
        v_exp = v.expand(-1, self.ACTION_NUM)                           # (B, ACTION_NUM)

        out = triton_combine(a, v_exp)                                 # (B, ACTION_NUM)
        return out