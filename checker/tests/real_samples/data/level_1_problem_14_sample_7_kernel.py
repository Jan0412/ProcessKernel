import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def upper_triangular_matmul_kernel(
    A, B, C,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Computes C = A @ B where A, B, and C are N x N upper triangular matrices.
    """
    # 1. Compute the program ID for the output tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. Create pointers to the beginning of the A and B blocks
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # 3. Initialize the accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 4. Loop over the K dimension
    for off_k in range(0, N, BLOCK_SIZE_K):
        # Create offsets for K
        offs_k = off_k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load A tile: A[i, k]
        # A is upper triangular, so A[i, k] is valid only if i <= k
        mask_a = (offs_am[:, None] <= offs_k[None, :]) & (offs_am[:, None] < N) & (offs_k[None, :] < N)
        a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load B tile: B[k, j]
        # B is upper triangular, so B[k, j] is valid only if k <= j
        mask_b = (offs_k[:, None] <= offs_bn[None, :]) & (offs_k[:, None] < N) & (offs_bn[None, :] < N)
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Accumulate the dot product
        accumulator += tl.dot(a, b)

    # 5. Mask the output for the upper triangle
    # C[i, j] is valid only if i <= j
    mask_c = (offs_am[:, None] <= offs_bn[None, :]) & (offs_am[:, None] < N) & (offs_bn[None, :] < N)
    
    # Create pointers to the output
    c_ptrs = C + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    
    # Store the result
    tl.store(c_ptrs, accumulator, mask=mask_c)


def triton_upper_tri_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper to launch the Triton kernel for upper triangular matrix multiplication.
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    assert A.shape == (N, N) and B.shape == (N, N)
    
    # Allocate output tensor (initialized to zeros)
    C = torch.zeros((N, N), device=A.device, dtype=A.dtype)
    
    # Configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Grid calculation
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        1
    )
    
    # Launch kernel
    upper_triangular_matmul_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices
    using a fused Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using Triton.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, which is upper triangular.
        """
        return triton_upper_tri_matmul(A, B)