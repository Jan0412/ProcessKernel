import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    A_ptr,  # Pointer to 1D diagonal tensor A
    B_ptr,  # Pointer to 2D matrix B
    out_ptr,  # Pointer to output matrix
    N,      # Number of rows in output (size of A)
    M,      # Number of columns in output (size of B columns)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get the block indices
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)
    
    # Compute the start indices for this block
    start_m = block_id_m * BLOCK_SIZE_M
    start_n = block_id_n * BLOCK_SIZE_N
    
    # Create offsets for the current block
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to ensure we don't go out of bounds
    mask_m = offsets_m < N
    mask_n = offsets_n < M
    
    # Load A values (broadcasting along rows)
    # A is 1D, so we load A[offsets_m] with proper masking
    a_offsets = offsets_m
    a_mask = mask_m
    a_vals = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
    
    # Load B values
    b_offsets = offsets_m[:, None] * M + offsets_n[None, :]
    b_mask = mask_m[:, None] & mask_n[None, :]
    b_vals = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
    
    # Perform element-wise multiplication
    out_vals = a_vals[:, None] * b_vals
    
    # Store the result
    out_offsets = offsets_m[:, None] * M + offsets_n[None, :]
    tl.store(out_ptr + out_offsets, out_vals, mask=b_mask)

def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized implementation of diag(A) @ B using Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], "Dimension mismatch between A and B"
    
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    N, M = A.shape[0], B.shape[1]
    out = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    
    # Calculate grid dimensions
    grid_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (M + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Launch the Triton kernel
    diag_matmul_kernel[grid](
        A, B, out, N, M, 
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)