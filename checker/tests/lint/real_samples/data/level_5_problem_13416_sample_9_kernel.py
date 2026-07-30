import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        cur_k = tl.minimum(BLOCK_K, K - k)
        offs_k = k + tl.arange(0, cur_k)

        a = tl.load(
            A + (offs_m[:, None] * K + offs_k[None, :]),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + (offs_k[:, None] * N + offs_n[None, :]),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    c = acc
    c_ptrs = C + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(
        c_ptrs,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def matmul_scaled_left_kernel(
    A, scale, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        cur_k = tl.minimum(BLOCK_K, K - k)
        offs_k = k + tl.arange(0, cur_k)

        a = tl.load(
            A + (offs_m[:, None] * K + offs_k[None, :]),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        s = tl.load(scale + offs_k, mask=offs_k < K, other=0.0)
        a = a * s[None, :]          # column‑wise scaling

        b = tl.load(
            B + (offs_k[:, None] * N + offs_n[None, :]),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    c = acc
    c_ptrs = C + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(
        c_ptrs,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# --------------------------------------------------------------
# Python wrappers around the kernels
# --------------------------------------------------------------

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Dense matrix multiplication A @ B using Triton."""
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


def triton_matmul_scaled_left(
    A: torch.Tensor, scale: torch.Tensor, B: torch.Tensor
) -> torch.Tensor:
    """
    Compute (A * scale) @ B where ``scale`` is a 1‑D tensor of length K
    that scales the columns of ``A``.
    """
    assert A.is_cuda and B.is_cuda and scale.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert scale.shape == (K,)

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )
    matmul_scaled_left_kernel[grid](
        A, scale, B, C,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


# --------------------------------------------------------------
# Optimized model
# --------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=False, K=10, alpha=0.1, **kwargs):
        super().__init__()
        assert K > 0
        self.K = K
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, U, V=None):
        """
        x: node attribute matrix (N, in_features)
        If V is None:
            U is the (N, N) adjacency matrix (dense or sparse)
        Else:
            U is (N, k) eigenvector matrix
            V is (k,) eigenvalue vector
        """
        x = self.w(x)                     # linear projection

        if V is not None:
            # ---- eigenvalue branch (dense) ----
            # compute V_out = (1-alpha) * sum_{i=1}^K V**i / K
            V_pow = V.clone()
            V_out = torch.zeros_like(V)
            for _ in range(self.K):
                V_pow = V_pow * V
                V_out += (1.0 - self.alpha) * V_pow
            V_out = V_out / self.K

            # M = U^T @ x  (k x out_features)
            M = triton_matmul(U.t().contiguous(), x.contiguous())

            # x_out = (U * V_out) @ M  (N x out_features)
            x_out = triton_matmul_scaled_left(
                U.contiguous(),
                V_out.contiguous(),
                M.contiguous(),
            )
            # residual connection
            x_out = x_out + self.alpha * x
            return x_out
        else:
            # ---- adjacency (sparse) branch ----
            # Keep the original implementation; it already uses efficient
            # sparse matrix multiplication.
            adj = U
            x_in = x
            x_out = torch.zeros_like(x)
            for _ in range(self.K):
                x = torch.spmm(adj, x)
                x_out += (1.0 - self.alpha) * x
            x_out = x_out / self.K
            x_out = x_out + self.alpha * x_in
            return x_out

    def reset_parameters(self):
        self.w.reset_parameters()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_features}, {self.out_features}, "
            f"K={self.K}, alpha={self.alpha})"
        )