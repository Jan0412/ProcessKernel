import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_lower_tril_kernel(
    A_ptr,  # Pointer to matrix A (row-major)
    B_ptr,  # Pointer to matrix B (row-major)
    C_ptr,  # Pointer to output matrix C (row-major)
    M,  # Number of rows in A and C
    N,  # Number of columns in B and C
    K,  # Number of columns in A and rows in B
    stride_am, stride_ak,  # Strides for A
    stride_bk, stride_bn,  # Strides for B
    stride_cm, stride_cn,  # Strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D block index
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # We only need to compute the lower triangular part of the output.
    # If the block is strictly in the upper triangle (block row < block col), skip it.
    if pid_m < pid_n:
        return

    # Offsets for the current block of C
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create a mask for the lower triangular part within the block
    # We only want to compute/store where row_idx >= col_idx
    mask = (off_m[:, None] >= off_n[None, :])

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for pid_k in range(0, tl.cdiv(K, BLOCK_K)):
        # Offsets for A and B
        off_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load A tile (BLOCK_M x BLOCK_K)
        # A is at [off_m, off_k]
        a_mask = (off_m[:, None] < M) & (off_k[None, :] < K)
        a = tl.load(A_ptr + off_m[:, None] * stride_am + off_k[None, :] * stride_ak, 
                    mask=a_mask, other=0.0)
        
        # Load B tile (BLOCK_K x BLOCK_N)
        # B is at [off_k, off_n]
        b_mask = (off_k[:, None] < K) & (off_n[None, :] < N)
        b = tl.load(B_ptr + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn, 
                    mask=b_mask, other=0.0)
        
        # Perform matrix multiplication accumulation
        acc = tl.dot(a, b) + acc

    # Store the result to C
    # Apply the lower triangular mask
    tl.store(C_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn, 
             acc, mask=mask)


def triton_matmul_lower_tril(A: torch.Tensor, B: torch.Tensor):
    """
    Computes the lower triangular part of the matrix product A @ B using Triton.
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Output tensor initialized to zeros (torch.tril behavior)
    C = torch.zeros_like(A)
    
    M, K = A.shape
    KB, N = B.shape
    assert K == KB, "Inner dimensions must match"
    
    # Determine block sizes (tunable)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    
    # Grid configuration
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M, 
        (N + BLOCK_N - 1) // BLOCK_N
    )
    
    # Launch kernel
    matmul_lower_tril_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul_lower_tril(A, B)