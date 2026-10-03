import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _layer_norm_kernel(
    X, W, B, Out,
    stride_x_b, stride_x_f, stride_x_n,
    stride_w_f, stride_w_n,
    stride_b_f, stride_b_n,
    stride_o_b, stride_o_f, stride_o_n,
    B, F, N, eps,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for Layer Normalization.
    Normalizes over the last dimension N, which corresponds to dim1 * dim2.
    Grid is over (B, F).
    """
    pid_b = tl.program_id(0)
    pid_f = tl.program_id(1)

    # Calculate base pointers for the current (batch, feature) slice
    x_ptr = X + pid_b * stride_x_b + pid_f * stride_x_f
    w_ptr = W + pid_f * stride_w_f
    b_ptr = B + pid_f * stride_b_f
    out_ptr = Out + pid_b * stride_o_b + pid_f * stride_o_f

    # Initialize accumulators for mean and variance
    sum_x = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum_xsq = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Offsets within the block
    off_n = tl.arange(0, BLOCK_N)

    # Pass 1: Compute mean and variance
    for start_n in range(0, N, BLOCK_N):
        n_idx = start_n + off_n
        mask = n_idx < N

        x = tl.load(x_ptr + n_idx * stride_x_n, mask=mask, other=0.0)

        sum_x += x
        sum_xsq += x * x

    # Reduce sums to scalars
    mean = tl.sum(sum_x) / N
    var = tl.sum(sum_xsq) / N - mean * mean

    # Compute inverse square root for normalization
    rsqrt = 1.0 / tl.sqrt(var + eps)

    # Pass 2: Normalize and apply affine transformation
    for start_n in range(0, N, BLOCK_N):
        n_idx = start_n + off_n
        mask = n_idx < N

        x = tl.load(x_ptr + n_idx * stride_x_n, mask=mask, other=0.0)

        # Load weight and bias for the current element
        w = tl.load(w_ptr + n_idx * stride_w_n, mask=mask, other=0.0)
        b = tl.load(b_ptr + n_idx * stride_b_n, mask=mask, other=0.0)

        # Apply normalization: (x - mean) * rsqrt * w + b
        out = (x - mean) * rsqrt * w + b

        tl.store(out_ptr + n_idx * stride_o_n, out, mask=mask)


def triton_layer_norm(x, w, b, eps):
    """
    Wrapper function to launch the Triton LayerNorm kernel.
    """
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    B, F, D1, D2 = x.shape
    N = D1 * D2

    out = torch.empty_like(x)

    # Grid dimensions: one program per (batch, feature) pair
    grid = (B, F)
    BLOCK_N = 256  # Tile size for the inner dimensions

    _layer_norm_kernel[grid](
        x, w, b, out,
        F*D1*D2, D1*D2, 1,  # X strides
        D1*D2, 1,          # W strides
        D1*D2, 1,          # B strides
        F*D1*D2, D1*D2, 1, # Out strides
        B, F, N, eps,
        BLOCK_N=BLOCK_N
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom Triton kernels.
    """
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters from the standard LayerNorm module
        w = self.ln.weight
        b = self.ln.bias
        eps = self.ln.eps

        # Call the optimized Triton kernel
        return triton_layer_norm(x, w, b, eps)