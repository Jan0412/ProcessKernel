import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    A_ptr,  # Pointer to diagonal vector (N,)
    B_ptr,  # Pointer to second matrix (N, M)
    C_ptr,  # Pointer to output matrix (N, M)
    N,      # Number of rows in B and C
    M,      # Number of columns in B and C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get the row index for this block
    row_block_id = tl.program_id(0)
    # Get the column index for this block
    col_block_id = tl.program_id(1)
    
    # Calculate the starting row and column for this block
    row_start = row_block_id * BLOCK_SIZE_M
    col_start = col_block_id * BLOCK_SIZE_N
    
    # Create offsets for the current block
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for valid indices
    row_mask = row_offsets < N
    col_mask = col_offsets < M
    
    # Combine masks for 2D access
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Load A values (broadcast to all columns)
    a_offsets = row_offsets
    a_mask = row_mask
    a_vals = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
    
    # Load B values
    b_offsets = row_offsets[:, None] * M + col_offsets[None, :]
    b_vals = tl.load(B_ptr + b_offsets, mask=mask, other=0.0)
    
    # Perform element-wise multiplication
    c_vals = a_vals[:, None] * b_vals
    
    # Store the result
    c_offsets = row_offsets[:, None] * M + col_offsets[None, :]
    tl.store(C_ptr + c_offsets, c_vals, mask=mask)

def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs diagonal matrix multiplication using Triton kernel.
    C = diag(A) @ B
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], "Dimension mismatch between A and B"
    
    # Prepare output tensor
    N, M = B.shape
    C = torch.empty_like(B)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    
    # Determine grid dimensions
    grid_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (M + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Launch the Triton kernel
    diag_matmul_kernel[grid](
        A, B, C, N, M, 
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)