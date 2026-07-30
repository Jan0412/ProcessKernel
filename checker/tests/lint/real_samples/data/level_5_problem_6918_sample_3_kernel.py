import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import List

# --------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------

@triton.jit
def exp_transpose_kernel(
    out_ptr,          # [B, K] original similarity matrix
    q_ptr,            # [K, B] output matrix (pre‑norm)
    B, K,
    eps,
    BLOCK_M: tl.constexpr,  # rows of out (B)
    BLOCK_N: tl.constexpr,  # cols of out (K)
):
    pid_m = tl.program_id(0)  # block over B
    pid_n = tl.program_id(1)  # block over K

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < B
    mask_n = offs_n < K

    # Load a tile of out, compute exp(out/eps) and write transposed
    out = tl.load(out_ptr + offs_m[:, None] * K + offs_n[None, :],
                  mask=mask_m[:, None] & mask_n[None, :],
                  other=0.0)
    q = tl.exp(out / eps)

    # write transposed tile to q (shape K x B)
    tl.store(q_ptr + offs_n[:, None] * B + offs_m[None, :],
             q,
             mask=mask_n[:, None] & mask_m[None, :])


def triton_exp_transpose(out: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute Q = exp(out/eps).t() using a Triton kernel."""
    B, K = out.shape
    q = torch.empty((K, B), dtype=out.dtype, device=out.device)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = ( (B + BLOCK_M - 1) // BLOCK_M,
             (K + BLOCK_N - 1) // BLOCK_N )
    exp_transpose_kernel[grid](
        out,
        q,
        B, K,
        eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return q


@triton.jit
def sinkhorn_iter_kernel(
    q_ptr,          # [K, B] matrix to be normalized in‑place
    row_sum_ptr,    # [K, 1] temporary row sums
    col_sum_ptr,    # [1, B] temporary column sums
    K, B,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_k = tl.program_id(0)  # over K
    pid_b = tl.program_id(1)  # over B

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    mask_k = offs_k < K
    mask_b = offs_b < B

    # ------------------------------------------------------------------
    # 1) Row sums  (sum over B for each k)
    # ------------------------------------------------------------------
    # Load a tile
    tile = tl.load(q_ptr + offs_k[:, None] * B + offs_b[None, :],
                   mask=mask_k[:, None] & mask_b[None, :],
                   other=0.0)
    row_sum = tl.sum(tile, axis=1)  # shape [BLOCK_K]

    # Write row sums (broadcasted column dimension)
    tl.store(row_sum_ptr + offs_k, row_sum, mask=mask_k)

    # ------------------------------------------------------------------
    # 2) Scale rows: q = q / row_sum
    # ------------------------------------------------------------------
    row_sum_tile = tl.load(row_sum_ptr + offs_k, mask=mask_k, other=1.0)
    row_sum_tile = row_sum_tile[:, None]  # [BLOCK_K,1]
    tile = tile / row_sum_tile

    # ------------------------------------------------------------------
    # 3) Column sums (sum over K for each b)
    # ------------------------------------------------------------------
    col_sum = tl.sum(tile, axis=0)  # shape [BLOCK_B]
    tl.store(col_sum_ptr + offs_b, col_sum, mask=mask_b)

    # ------------------------------------------------------------------
    # 4) Scale columns: q = q / col_sum
    # ------------------------------------------------------------------
    col_sum_tile = tl.load(col_sum_ptr + offs_b, mask=mask_b, other=1.0)
    col_sum_tile = col_sum_tile[None, :]  # [1, BLOCK_B]
    tile = tile / col_sum_tile

    # ------------------------------------------------------------------
    # 5) Write back normalized tile
    # ------------------------------------------------------------------
    tl.store(q_ptr + offs_k[:, None] * B + offs_b[None, :],
             tile,
             mask=mask_k[:, None] & mask_b[None, :])


def triton_sinkhorn(q: torch.Tensor, iterations: int, K: int, B: int) -> torch.Tensor:
    """Perform the Sinkhorn iterations on Q = exp(out/eps).t() in‑place."""
    # temporary buffers for row/col sums
    row_sum = torch.empty(K, dtype=q.dtype, device=q.device)
    col_sum = torch.empty(B, dtype=q.dtype, device=q.device)

    BLOCK_K = 64
    BLOCK_B = 64
    grid = ( (K + BLOCK_K - 1) // BLOCK_K,
             (B + BLOCK_B - 1) // BLOCK_B )

    for _ in range(iterations):
        sinkhorn_iter_kernel[grid](
            q,
            row_sum,
            col_sum,
            K, B,
            BLOCK_K=BLOCK_K,
            BLOCK_B=BLOCK_B,
        )
        # the algorithm also multiplies by 1/K and 1/B inside the loop;
        # we fold these constants into the next iteration by scaling once
        q.mul_(1.0 / K)
        q.mul_(1.0 / B)

    # after the loop the reference implementation multiplies by B
    q.mul_(B)
    return q


@triton.jit
def log_softmax_kernel(
    z_ptr,          # [B, K]
    out_ptr,        # [B, K] = log_softmax(z / T)
    B, K,
    T,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_b = offs_b < B
    mask_k = offs_k < K

    # Load tile
    z = tl.load(z_ptr + offs_b[:, None] * K + offs_k[None, :],
                mask=mask_b[:, None] & mask_k[None, :],
                other=-float('inf'))

    # Scale
    z = z / T

    # Row‑wise max
    row_max = tl.max(z, axis=1)[:, None]
    z_shift = z - row_max

    # exp and sum
    exp_z = tl.exp(z_shift)
    row_sum = tl.sum(exp_z, axis=1)[:, None]
    log_sum_exp = tl.log(row_sum)

    # log‑softmax
    out = z_shift - log_sum_exp

    tl.store(out_ptr + offs_b[:, None] * K + offs_k[None, :],
             out,
             mask=mask_b[:, None] & mask_k[None, :])


def triton_log_softmax(z: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute log_softmax(z / temperature) with a Triton kernel."""
    B, K = z.shape
    out = torch.empty_like(z)

    BLOCK_B = 64
    BLOCK_K = 64
    grid = ( (B + BLOCK_B - 1) // BLOCK_B,
             (K + BLOCK_K - 1) // BLOCK_K )
    log_softmax_kernel[grid](
        z,
        out,
        B, K,
        temperature,
        BLOCK_B=BLOCK_B,
        BLOCK_K=BLOCK_K,
    )
    return out


@triton.jit
def subloss_kernel(
    q_ptr,          # [B, K]  (already transposed to match z)
    logz_ptr,       # [B, K] = log_softmax(z / T)
    out_ptr,        # [] scalar (float) – will be written atomically
    B, K,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < B * K

    # flatten indices
    q = tl.load(q_ptr + offs, mask=mask, other=0.0)
    logz = tl.load(logz_ptr + offs, mask=mask, other=0.0)

    # contribution = - q * logz
    contrib = - q * logz

    # reduction within the block
    block_sum = tl.sum(contrib)

    # atomic add to the scalar output (using 32‑bit float)
    tl.atomic_add(out_ptr, block_sum)


def triton_subloss(z: torch.Tensor, q: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Compute -mean( sum_i q_i * log_softmax(z_i / T) ) using Triton.
    z : [B, K]
    q : [B, K]   (already matching shape)
    """
    B, K = z.shape
    logz = triton_log_softmax(z, temperature)

    # allocate a single‑element tensor for the reduction result
    out = torch.zeros(1, dtype=z.dtype, device=z.device)

    BLOCK = 1024
    grid = ( (B * K + BLOCK - 1) // BLOCK, )
    subloss_kernel[grid](
        q,
        logz,
        out,
        B, K,
        BLOCK=BLOCK,
    )
    # mean over batch dimension
    return out / B


# --------------------------------------------------------------
# Optimized Model
# --------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Optimized version of the SwaV loss with Triton kernels for
    - the Sinkhorn normalisation,
    - log‑softmax,
    - the sub‑loss reduction.
    """
    def __init__(self, temperature: float = 0.1,
                 sinkhorn_iterations: int = 3,
                 sinkhorn_epsilon: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon

    def _sinkhorn_triton(self, out: torch.Tensor) -> torch.Tensor:
        """
        out : [B, K] similarity matrix (float32, CUDA)
        Returns Q : [B, K] soft assignment matrix (same shape as input).
        """
        B, K = out.shape
        # 1) exp(out / eps) and transpose -> shape [K, B]
        q = triton_exp_transpose(out, self.sinkhorn_epsilon)
        # 2) Run Sinkhorn iterations (in‑place on the transposed matrix)
        q = triton_sinkhorn(q, self.sinkhorn_iterations, K, B)
        # 3) Transpose back to [B, K] to match the rest of the code
        return q.t()

    def _subloss_triton(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross‑entropy term using Triton.
        """
        return triton_subloss(z, q, self.temperature)

    def forward(self,
                high_resolution_outputs: List[torch.Tensor],
                low_resolution_outputs: List[torch.Tensor]) -> torch.Tensor:
        n_crops = len(high_resolution_outputs) + len(low_resolution_outputs)
        loss = 0.0

        for i in range(len(high_resolution_outputs)):
            # Sinkhorn on a detached copy (no gradient through it)
            with torch.no_grad():
                q = self._sinkhorn_triton(high_resolution_outputs[i].detach())

            subloss = 0.0
            # compare with every other high‑resolution view
            for v in range(len(high_resolution_outputs)):
                if v != i:
                    subloss += self._subloss_triton(high_resolution_outputs[v], q)
            # and with all low‑resolution views
            for v in range(len(low_resolution_outputs)):
                subloss += self._subloss_triton(low_resolution_outputs[v], q)

            loss += subloss / (n_crops - 1)

        return loss / len(high_resolution_outputs)


# --------------------------------------------------------------
# Helper functions (kept for compatibility with the original script)
# --------------------------------------------------------------

def get_inputs():
    # Example shapes used by the original benchmark – they are arbitrary.
    return [torch.rand([4, 4, 4], device='cuda'),
            torch.rand([4, 4, 4], device='cuda')]


def get_init_inputs():
    return []


# Alias expected by the benchmarking harness
Model = ModelNew