import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------------

@triton.jit
def diag_sqrt_kernel(
    cov_ptr,          # pointer to covariance matrix
    out_ptr,          # pointer to stds output (B, N)
    n_elements,       # total number of diagonal elements = B * N
    n_assets: tl.constexpr,   # number of assets (N)
    BLOCK_SIZE: tl.constexpr,
):
    # each program works on a contiguous block of diagonal entries
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # linear index -> (batch, asset)
    batch = offsets // n_assets
    asset = offsets % n_assets

    # linear index of the diagonal element in the (B,N,N) tensor
    idx = batch * n_assets * n_assets + asset * n_assets + asset

    cov_val = tl.load(cov_ptr + idx, mask=mask, other=0.0)
    std_val = tl.sqrt(cov_val)

    tl.store(out_ptr + offsets, std_val, mask=mask)


@triton.jit
def corr_kernel(
    cov_ptr,          # pointer to covariance matrix (B,N,N)
    std_ptr,          # pointer to stds (B,N)
    out_ptr,          # pointer to correlation matrix (B,N,N)
    n_samples: tl.constexpr,   # B
    n_assets: tl.constexpr,    # N
    total_elements: tl.constexpr,   # B*N*N
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    n_sq = n_assets * n_assets

    # decode (b,i,j) from linear offset
    b = offsets // n_sq
    rem = offsets % n_sq
    i = rem // n_assets
    j = rem % n_assets

    # load values
    cov = tl.load(cov_ptr + offsets, mask=mask, other=0.0)

    std_i = tl.load(std_ptr + b * n_assets + i, mask=mask, other=1.0)
    std_j = tl.load(std_ptr + b * n_assets + j, mask=mask, other=1.0)

    corr = cov / (std_i * std_j)

    tl.store(out_ptr + offsets, corr, mask=mask)


# ------------------------------------------------------------------
# Helper that dispatches the Triton kernels
# ------------------------------------------------------------------
def triton_cov2corr(covmat: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of covariance matrices to correlation matrices using
    custom Triton kernels.
    Expected shape: (B, N, N), dtype=torch.float32, device='cuda'.
    """
    assert covmat.is_cuda, "Input must be a CUDA tensor"
    assert covmat.dtype == torch.float32, "Only FP32 is supported"

    cov = covmat.contiguous()
    B, N, _ = cov.shape

    # ------------------------------------------------------------------
    # 1) Compute per‑sample standard deviations (sqrt of diagonal)
    # ------------------------------------------------------------------
    stds = torch.empty((B, N), dtype=cov.dtype, device=cov.device)

    total_diag = B * N
    BLOCK = 128  # tunable
    grid_diag = lambda meta: ((total_diag + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    diag_sqrt_kernel[grid_diag](
        cov,
        stds,
        total_diag,
        n_assets=N,
        BLOCK_SIZE=BLOCK,
    )

    # ------------------------------------------------------------------
    # 2) Compute correlation matrix = cov / (std_i * std_j)
    # ------------------------------------------------------------------
    out = torch.empty_like(cov)

    total = B * N * N
    grid_corr = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    corr_kernel[grid_corr](
        cov,
        stds,
        out,
        n_samples=B,
        n_assets=N,
        total_elements=total,
        BLOCK_SIZE=BLOCK,
    )

    return out


# ------------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """Optimized Cov2Corr using Triton kernels."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, covmat: torch.Tensor) -> torch.Tensor:
        return triton_cov2corr(covmat)