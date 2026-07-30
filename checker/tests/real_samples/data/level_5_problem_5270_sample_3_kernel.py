import torch
import triton
import triton.language as tl


# --------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------

@triton.jit
def row_norm_sq_kernel(
    X_ptr,            # *float32, input matrix (N, D)
    out_ptr,          # *float32, output vector (N,)
    N, D,             # int32, dimensions
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_row = row < N

    # accumulator for each row
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # loop over columns
    for d in range(0, D, BLOCK_D):
        col = d + tl.arange(0, BLOCK_D)
        mask = (col < D) & mask_row[:, None]

        # load a tile of X
        x = tl.load(X_ptr + row[:, None] * D + col, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=1)

    # write result
    tl.store(out_ptr + row, acc, mask=mask_row)


def triton_row_norm_sq(X: torch.Tensor) -> torch.Tensor:
    """Compute squared L2‑norm of each row of X using Triton."""
    assert X.is_cuda
    X = X.contiguous()
    N, D = X.shape
    out = torch.empty(N, dtype=torch.float32, device=X.device)

    BLOCK_N = 128
    BLOCK_D = 64
    grid = lambda meta: ((N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)

    row_norm_sq_kernel[grid](
        X,
        out,
        N,
        D,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out


@triton.jit
def distance_kernel(
    dot_ptr,          # *float32, X @ Y^T  (M, N)
    xnorm_ptr,        # *float32, ||x_i||^2  (M,)
    ynorm_ptr,        # *float32, ||y_j||^2  (N,)
    out_ptr,          # *float32, distance matrix (M, N)
    M, N,             # int32 dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = row < M
    mask_n = col < N

    # load dot products
    dot = tl.load(
        dot_ptr + row[:, None] * N + col,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    # load norms
    xnorm = tl.load(xnorm_ptr + row, mask=mask_m, other=0.0)
    ynorm = tl.load(ynorm_ptr + col, mask=mask_n, other=0.0)

    # compute squared Euclidean distance
    dist = xnorm[:, None] + ynorm[None, :] - 2.0 * dot

    tl.store(
        out_ptr + row[:, None] * N + col,
        dist,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def exp_kernel(
    dist_ptr,         # *float32, distance matrix (M, N)
    out_ptr,          # *float32, output kernel (M, N)
    M, N, gamma,      # int32, int32, float32
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = row < M
    mask_n = col < N

    d = tl.load(
        dist_ptr + row[:, None] * N + col,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    out = tl.exp(-gamma * d)

    tl.store(
        out_ptr + row[:, None] * N + col,
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def triton_distance(dot: torch.Tensor,
                    xnorm: torch.Tensor,
                    ynorm: torch.Tensor) -> torch.Tensor:
    """Fused distance computation (no exponential)."""
    M, N = dot.shape
    out = torch.empty_like(dot)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    distance_kernel[grid](
        dot,
        xnorm,
        ynorm,
        out,
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out


def triton_exp(dist: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply exp(-gamma * dist) element‑wise."""
    M, N = dist.shape
    out = torch.empty_like(dist)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    exp_kernel[grid](
        dist,
        out,
        M,
        N,
        gamma,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out


# --------------------------------------------------------------
# Optimized RBF kernel using the Triton primitives above
# --------------------------------------------------------------

class ModelNew(torch.nn.Module):
    """
    Optimized RBF kernel (FP32) with Triton kernels.
    """

    def __init__(self, bandwidth=None):
        super().__init__()
        self.bandwidth = bandwidth

    def _bandwidth(self, norm_sq: torch.Tensor):
        if self.bandwidth is None:
            # median on GPU to avoid host transfer
            med = torch.median(norm_sq).item()
            h = med / (2.0 * torch.log(torch.tensor(norm_sq.shape[0] + 1.0, device=norm_sq.device)))
            return torch.sqrt(h).item()
        else:
            return self.bandwidth

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        X : (M, D)
        Y : (N, D)
        returns K : (M, N)
        """
        assert X.is_cuda and Y.is_cuda, "Both inputs must be on CUDA."
        X = X.contiguous()
        Y = Y.contiguous()

        # 1) row norms
        xnorm = triton_row_norm_sq(X)          # (M,)
        ynorm = triton_row_norm_sq(Y)          # (N,)

        # 2) dot product
        dot = torch.matmul(X, Y.t())           # (M, N)

        # 3) squared Euclidean distances
        dnorm2 = triton_distance(dot, xnorm, ynorm)   # (M, N)

        # 4) bandwidth & gamma
        bandwidth = self._bandwidth(dnorm2)
        gamma = 1.0 / (1e-8 + 2.0 * bandwidth ** 2)

        # 5) final RBF kernel
        K = triton_exp(dnorm2, gamma)          # (M, N)
        return K