import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_min_kernel(
    X, W, Y, min_vals,
    stride_xb, stride_xc,
    stride_wc, stride_wc,
    stride_yb, stride_yc,
    B, IC, OC,
    BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Computes Y = X @ W.T and min_vals = min(Y, dim=1).
    """
    # 2D block indexing
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Offsets
    off_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Initialize accumulator for the dot product
    acc = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)

    # Loop over the IC dimension
    for n in range(0, IC, BLOCK_N):
        # Load X: [BLOCK_B, BLOCK_N]
        x_ptrs = X + off_b[:, None] * stride_xb + (n + tl.arange(0, BLOCK_N)[None, :]) * stride_xc
        x = tl.load(x_ptrs, mask=((off_b[:, None] < B) & (tl.arange(0, BLOCK_N)[None, :] < IC)), other=0.0)

        # Load W: [BLOCK_N, BLOCK_M] (W is stored as [OC, IC], we want [IC, OC])
        # W.T[IC, OC] -> W[OC, IC]
        w_ptrs = W + (n + tl.arange(0, BLOCK_N)[:, None]) * stride_wc + off_m[None, :] * stride_wc
        w = tl.load(w_ptrs, mask=((tl.arange(0, BLOCK_N)[:, None] < IC) & (off_m[None, :] < OC)), other=0.0)

        # Accumulate
        acc += tl.dot(x, w)

    # Store Y
    y_ptrs = Y + off_b[:, None] * stride_yb + off_m[None, :] * stride_yc
    tl.store(y_ptrs, acc, mask=((off_b[:, None] < B) & (off_m[None, :] < OC)))

    # Compute local min for this block and update global min_vals using atomics
    # We need to find the min across the BLOCK_M dimension for each row 'b'
    local_min = tl.min(acc, axis=1)

    # We only update if this block is valid for the batch
    mask_b = off_b < B

    # Use atomic min to update the global min_vals array
    # Note: Triton's atomic_min works on individual elements. We can loop over BLOCK_B or use a reduction.
    # Since BLOCK_B is likely small, let's just use a simple loop or broadcast if BLOCK_B=1.
    # To keep it simple and robust, we will just launch enough blocks and use atomic_min.
    # We need to handle the mask correctly.
    min_ptrs = min_vals + off_b
    tl.atomic_min(min_ptrs, local_min, mask=mask_b)


@triton.jit
def group_norm_min_bias_kernel(
    Y, Out, min_vals, bias, gamma, beta,
    stride_yb, stride_yc,
    stride_ob, stride_oc,
    B, OC, GROUP_SIZE,
    BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr
):
    """
    Performs GroupNorm, subtracts min_vals, and adds bias.
    GroupNorm normalizes across the group dimension (part of the channel dimension).
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    off_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Load Y tile
    y_ptrs = Y + off_b[:, None] * stride_yb + off_m[None, :] * stride_yc
    y = tl.load(y_ptrs, mask=((off_b[:, None] < B) & (off_m[None, :] < OC)), other=0.0)

    # Determine group index for each column in the block
    # group_id = col // GROUP_SIZE
    group_ids = off_m // GROUP_SIZE

    # We need to compute mean and variance for each group within this tile.
    # Because GROUP_SIZE is small (16) and BLOCK_M is likely >= GROUP_SIZE, 
    # we can compute exact statistics locally within the block.

    # Initialize sums per group
    # Max groups in a block is BLOCK_M // GROUP_SIZE + 1
    num_groups_in_block = tl.cdiv(BLOCK_M, GROUP_SIZE)

    sum_y = tl.zeros((num_groups_in_block,), dtype=tl.float32)
    sum_y_sq = tl.zeros((num_groups_in_block,), dtype=tl.float32)

    # Loop over the columns in the block to aggregate group statistics
    for i in range(BLOCK_M):
        g = i // GROUP_SIZE
        val = tl.load(y_ptrs + i, mask=(off_b < B), other=0.0) # This is a vector of size BLOCK_B
        sum_y = tl.where(i // GROUP_SIZE == g, sum_y + tl.sum(val, axis=0), sum_y)
        sum_y_sq = tl.where(i // GROUP_SIZE == g, sum_y_sq + tl.sum(val * val, axis=0), sum_y_sq)

    # Compute mean and variance
    mean = sum_y / GROUP_SIZE
    var = sum_y_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Normalize, subtract min, add bias
    out = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)

    for i in range(BLOCK_M):
        g = i // GROUP_SIZE
        val = tl.load(y_ptrs + i, mask=(off_b < B), other=0.0)

        # Normalize
        normalized = (val - mean[g]) * rstd[g]

        # Load gamma and beta
        gamma_val = tl.load(gamma + off_m[i], mask=(off_m[i] < OC), other=1.0)
        beta_val = tl.load(beta + off_m[i], mask=(off_m[i] < OC), other=0.0)

        # Apply affine transform
        normalized = normalized * gamma_val + beta_val

        # Subtract min_vals
        min_val = tl.load(min_vals + off_b, mask=(off_b < B), other=0.0)
        normalized = normalized - min_val

        # Add bias
        bias_val = tl.load(bias + off_m[i], mask=(off_m[i] < OC), other=0.0)
        normalized = normalized + bias_val

        out = tl.store(out + i, normalized) # This syntax is invalid in Triton, we need to accumulate properly

    # Let's rewrite the accumulation loop to be Triton compliant
    out = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)
    for i in range(BLOCK_M):
        g = i // GROUP_SIZE
        val = tl.load(y_ptrs + i, mask=(off_b < B), other=0.0)
        normalized = (val - mean[g]) * rstd[g]
        gamma_val = tl.load(gamma + off_m[i], mask=(off_m[i] < OC), other=1.0)
        beta_val = tl.load(beta + off_m[i], mask=(off_m[i] < OC), other=0.0)
        normalized = normalized * gamma_val + beta_val
        min_val = tl.load(min_vals + off_b, mask=(off_b < B), other=0.0)
        normalized = normalized - min_val
        bias_val = tl.load(bias + off_m[i], mask=(off_m[i] < OC), other=0.0)
        normalized = normalized + bias_val

        out = tl.where(tl.arange(0, BLOCK_B)[:, None] == tl.arange(0, BLOCK_B)[:, None], out + normalized[:, None], out) 
        # The above 'tl.where' logic is flawed for accumulation. Let's just use a direct store into a pre-allocated output buffer or use a simpler approach.

    # Simpler approach for the final accumulation:
    out_tile = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)
    for i in range(BLOCK_M):
        g = i // GROUP_SIZE
        val = tl.load(y_ptrs + i, mask=(off_b < B), other=0.0)
        normalized = (val - mean[g]) * rstd[g]
        gamma_val = tl.load(gamma + off_m[i], mask=(off_m[i] < OC), other=1.0)
        beta_val = tl.load(beta + off_m[i], mask=(off_m[i] < OC), other=0.0)
        normalized = normalized * gamma_val + beta_val
        min_val = tl.load(min_vals + off_b, mask=(off_b < B), other=0.0)
        normalized = normalized - min_val
        bias_val = tl.load(bias + off_m[i], mask=(off_m[i] < OC), other=0.0)
        normalized = normalized + bias_val

        out_tile = tl.where(i == tl.arange(0, BLOCK_M)[None, :], out_tile + normalized[:, None], out_tile)

    out_ptrs = Out + off_b[:, None] * stride_ob + off_m[None, :] * stride_oc
    tl.store(out_ptrs, out_tile, mask=((off_b[:, None] < B) & (off_m[None, :] < OC)))


def triton_forward(x, weight, bias, gamma, beta, num_groups):
    B, IC = x.shape
    OC, _ = weight.shape
    GROUP_SIZE = OC // num_groups

    # Initialize outputs
    matmul_out = torch.empty((B, OC), dtype=torch.float32, device=x.device)
    min_vals = torch.full((B,), float('inf'), dtype=torch.float32, device=x.device)
    out = torch.empty((B, OC), dtype=torch.float32, device=x.device)

    # Grids
    grid_mm = (triton.cdiv(B, 4), triton.cdiv(OC, 32))
    grid_ops = (triton.cdiv(B, 4), triton.cdiv(OC, 32))

    # Launch Matmul + Min kernel
    matmul_min_kernel[grid_mm](
        x, weight, matmul_out, min_vals,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        matmul_out.stride(0), matmul_out.stride(1),
        B, IC, OC,
        BLOCK_B=4, BLOCK_M=32, BLOCK_N=32
    )

    # Launch GroupNorm + Min + Bias kernel
    group_norm_min_bias_kernel[grid_ops](
        matmul_out, out, min_vals, bias, gamma, beta,
        matmul_out.stride(0), matmul_out.stride(1),
        out.stride(0), out.stride(1),
        B, OC, GROUP_SIZE,
        BLOCK_B=4, BLOCK_M=32
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Extract parameters
        weight = self.gemm.weight
        bias_param = self.bias
        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        num_groups = self.group_norm.num_groups

        # Call fused Triton kernels
        return triton_forward(x, weight, bias_param, gamma, beta, num_groups)