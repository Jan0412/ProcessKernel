import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def lower_triangular_matmul_kernel(
    A, B, C,
    N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Computes C = A * B, assuming A and B are lower triangular.
    Only computes and stores the lower triangular part of C.
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets into the matrices
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    
    # Initialize the accumulator for the output tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over the K dimension (reduction dimension)
    for k_start in range(0, N, BLOCK_K):
        # Load A tile: A[row_start + m, k_start + k]
        # A is lower triangular, so A[i, k] is zero if i < k
        k_curr = k_start + tl.arange(0, BLOCK_K)
        m_curr = row_start + tl.arange(0, BLOCK_M)
        
        # Mask for A: valid if m_curr >= k_curr
        mask_a = m_curr[:, None] >= k_curr[None, :]
        
        a_offsets = m_curr[:, None] * N + k_curr[None, :]
        a_tile = tl.load(A + a_offsets, mask=mask_a, other=0.0)
        
        # Load B tile: B[k_start + k, col_start + n]
        # B is lower triangular, so B[k, j] is zero if k < j
        n_curr = col_start + tl.arange(0, BLOCK_N)
        
        # Mask for B: valid if k_curr >= n_curr
        mask_b = k_curr[:, None] >= n_curr[None, :]
        
        b_offsets = k_curr[:, None] * N + n_curr[None, :]
        b_tile = tl.load(B + b_offsets, mask=mask_b, other=0.0)
        
        # Accumulate: acc += A_tile * B_tile
        acc += tl.dot(a_tile, b_tile)
        
    # Store the result to C
    # C is lower triangular, so C[i, j] is zero if i < j
    # We only store if row >= col
    m_store = row_start + tl.arange(0, BLOCK_M)
    n_store = col_start + tl.arange(0, BLOCK_N)
    
    mask_store = m_store[:, None] >= n_store[None, :]
    
    c_offsets = m_store[:, None] * N + n_store[None, :]
    tl.store(C + c_offsets, acc, mask=mask_store)


def triton_lower_triangular_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Wraps the Triton kernel call for lower triangular matrix multiplication.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Input matrices must be square."
    
    N = A.shape[0]
    
    # Output tensor initialized to zeros
    C = torch.zeros_like(A)
    
    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    # Grid configuration
    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )
    
    # Launch the kernel
    lower_triangular_matmul_kernel[grid](
        A, B, C,
        N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for lower triangular matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B
        using a custom Triton kernel.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return triton_lower_triangular_matmul(A, B)