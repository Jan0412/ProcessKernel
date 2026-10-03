import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_sub_kernel(
    x_ptr,  # Pointer to input X (B, K)
    w_ptr,  # Pointer to weights W (M, K)
    sub_ptr,  # Pointer to subtract parameter (M,)
    out_ptr,  # Pointer to output Y (B, M)
    stride_xb, stride_xk,  # Strides for X
    stride_wm, stride_wk,  # Strides for W
    stride_om, stride_om,  # Strides for Output (B, M)
    M, K,  # Dimensions: M = out_features, K = in_features
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # N is K in GEMM terms (reduction dim)
    BLOCK_SIZE_K: tl.constexpr,  # K is M in GEMM terms (output dim) -> Wait, naming convention:
    # Let's stick to: X (B, K), W (M, K). Output Y (B, M).
    # Reduction is over K.
    # BLOCK_SIZE_M corresponds to B dimension blocks.
    # BLOCK_SIZE_N corresponds to M dimension blocks.
    # BLOCK_SIZE_K corresponds to K dimension blocks.
):
    """
    Computes Y = X @ W.T - Sub
    X: (B, K)
    W: (M, K)
    Sub: (M,)
    Y: (B, M)
    """
    pid_m = tl.program_id(0)  # Corresponds to B dimension
    pid_n = tl.program_id(1)  # Corresponds to M dimension

    # Offsets for B and M
    off_b = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_m = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K (the reduction dimension)
    for k in range(0, K, BLOCK_SIZE_K):
        off_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Load X block: (BLOCK_B, BLOCK_K)
        # X is (B, K). Strides: stride_xb for B, stride_xk for K.
        x_ptrs = x_ptr + off_b[:, None] * stride_xb + off_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=(off_b[:, None] < B) & (off_k[None, :] < K), other=0.0)

        # Load W block: (BLOCK_M, BLOCK_K)
        # W is (M, K). Strides: stride_wm for M, stride_wk for K.
        # We need W[m, k].
        w_ptrs = w_ptr + off_m[:, None] * stride_wm + off_k[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=(off_m[:, None] < M) & (off_k[None, :] < K), other=0.0)

        # Matrix multiply accumulate: X (B, K) @ W.T (K, M) -> (B, M)
        # tl.dot expects (BLOCK_M, BLOCK_K) and (BLOCK_K, BLOCK_N)
        # Here, 'x' is (BLOCK_B, BLOCK_K). 'w' is (BLOCK_M, BLOCK_K).
        # We need to transpose 'w' logically or swap operands.
        # tl.dot(a, b) computes a @ b.
        # We want X @ W.T.
        # So we compute tl.dot(x, w.T).
        # In Triton, tl.dot(a, b) where a is (M, K) and b is (N, K) is not directly supported as dot.
        # Standard tl.dot(a, b) assumes a is (M, K) and b is (K, N).
        # We have x (B, K) and w (M, K).
        # We can do tl.dot(x, w) if we treat w as (K, M) by transposing the layout in memory? No.
        # We must use tl.dot(x, w, trans_b=True).
        accumulator += tl.dot(x, w, trans_b=True)

    # Subtract the bias/subtract parameter
    # Sub is (M,) effectively, broadcasted to (BLOCK_B, BLOCK_M)
    sub_ptrs = sub_ptr + off_m
    sub = tl.load(sub_ptrs, mask=off_m < M, other=0.0)
    # Broadcast sub to (BLOCK_B, BLOCK_M)
    sub = sub[None, :]
    out = accumulator - sub

    # Store result
    # Output is (B, M)
    out_ptrs = out_ptr + off_b[:, None] * stride_om + off_m[None, :] * stride_om
    tl.store(out_ptrs, out, mask=(off_b[:, None] < B) & (off_m[None, :] < M))


def triton_matmul_sub(x, w, subtract):
    """
    Wrapper for matmul_sub_kernel.
    x: (B, K)
    w: (M, K) -> Weights from nn.Linear (not transposed)
    subtract: (M,)
    """
    assert x.is_cuda and w.is_cuda and subtract.is_cuda

    B, K = x.shape
    M = w.shape[0]

    # Output tensor
    out = torch.empty((B, M), device=x.device, dtype=x.dtype)

    # Block sizes
    BLOCK_SIZE_M = 128  # For B dimension
    BLOCK_SIZE_N = 128  # For M dimension
    BLOCK_SIZE_K = 32   # For K dimension

    # Grid
    grid = (
        triton.cdiv(B, BLOCK_SIZE_M),
        triton.cdiv(M, BLOCK_SIZE_N),
        1
    )

    matmul_sub_kernel[grid](
        x, w, subtract, out,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        M, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return out


@triton.jit
def mean_gelu_add_kernel(
    y_ptr,      # Pointer to intermediate output Y (B, M)
    x_orig_ptr, # Pointer to original input X (B, K)
    out_ptr,    # Pointer to final output (B, K)
    stride_yb, stride_ym,
    stride_xb, stride_xk,
    stride_ob, stride_ok,
    B, M, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Computes Out[b, :] = X_orig[b, :] + GELU(Mean(Y[b, :]))
    Each program instance handles one row (batch item).
    """
    pid = tl.program_id(0)

    if pid >= B:
        return

    # --- Step 1: Compute Mean of Y[b, :] ---
    row_sum = 0.0
    num_steps_m = tl.cdiv(M, BLOCK_SIZE_M)

    for i in range(num_steps_m):
        offset = i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask = offset < M
        y_vals = tl.load(y_ptr + pid * stride_yb + offset * stride_ym, mask=mask, other=0.0)
        row_sum += tl.sum(y_vals)

    mean_val = row_sum / M

    # --- Step 2: Apply GELU ---
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    SQRT_2_OVER_PI = 0.79788456
    GELU_COEFF = 0.044715

    mean_cubed = mean_val * mean_val * mean_val
    tanh_arg = SQRT_2_OVER_PI * (mean_val + GELU_COEFF * mean_cubed)
    gelu_val = 0.5 * mean_val * (1.0 + tl.math.tanh(tanh_arg))

    # --- Step 3: Add to Original X and Store ---
    num_steps_k = tl.cdiv(K, BLOCK_SIZE_K)
    x_row_start = x_orig_ptr + pid * stride_xb
    out_row_start = out_ptr + pid * stride_ob

    for i in range(num_steps_k):
        offset = i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask = offset < K

        x_vals = tl.load(x_row_start + offset * stride_xk, mask=mask, other=0.0)
        out_vals = x_vals + gelu_val

        tl.store(out_row_start + offset * stride_ok, out_vals, mask=mask)


def triton_mean_gelu_add(y, x_orig):
    """
    Wrapper for mean_gelu_add_kernel.
    y: (B, M)
    x_orig: (B, K)
    """
    assert y.is_cuda and x_orig.is_cuda

    B, M = y.shape
    K = x_orig.shape[1]

    out = torch.empty_like(x_orig)

    BLOCK_SIZE_M = 256  # Block size for reduction over M
    BLOCK_SIZE_K = 256  # Block size for write over K

    grid = (B,)

    mean_gelu_add_kernel[grid](
        y, x_orig, out,
        y.stride(0), y.stride(1),
        x_orig.stride(0), x_orig.stride(1),
        out.stride(0), out.stride(1),
        B, M, K,
        BLOCK_SIZE_M, BLOCK_SIZE_K
    )
    return out


class Model(nn.Module):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.clone().detach()
        # Gemm
        x = self.gemm(x)

        # Subtract
        x = x - self.subtract

        # GlobalAvgPool
        x = torch.mean(x, dim=1, keepdim=True)

        # LogSumExp
        x = torch.logsumexp(x, dim=1, keepdim=True)

        # GELU
        x = torch.nn.functional.gelu(x)

        # ResidualAdd
        x = x + original_x

        return x


class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # Keep a reference to original x for residual add. No clone needed as we don't modify x in place.
        original_x = x

        # 1. Matmul + Subtract
        # Pass weights directly without transposing or copying.
        y = triton_matmul_sub(x, self.gemm.weight, self.subtract)

        # 2. Mean + GELU + ResidualAdd
        # LogSumExp is identity after Mean on (B, 1) tensor, so skipped.
        out = triton_mean_gelu_add(y, original_x)

        return out


batch_size = 2048
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]