import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    A_ptr,  # Pointer to diagonal elements (1D)
    B_ptr,  # Pointer to matrix B (2D)
    C_ptr,  # Pointer to output matrix C (2D)
    N,      # Number of rows in B
    M,      # Number of columns in B
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the row index for this program
    row = tl.program_id(0)
    
    # Each program handles one row
    if row >= N:
        return
        
    # Load the diagonal element for this row
    a = tl.load(A_ptr + row)
    
    # Process multiple columns in this block
    for col in range(0, M, BLOCK_SIZE):
        # Calculate column index
        col_idx = col + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < M
        
        # Load B[row, col:col+BLOCK_SIZE]
        b = tl.load(B_ptr + row * M + col_idx, mask=mask, other=0.0)
        
        # Compute C[row, col:col+BLOCK_SIZE] = A[row] * B[row, col:col+BLOCK_SIZE]
        c = a * b
        
        # Store result
        tl.store(C_ptr + row * M + col_idx, c, mask=mask)

def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Custom Triton kernel for diagonal matrix multiplication: C = diag(A) @ B
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], "Dimension mismatch between A and B"
    
    N, M = B.shape
    
    # Prepare output tensor
    C = torch.empty_like(B)
    
    # Configure block size
    BLOCK_SIZE = 128
    
    # Determine grid size
    grid = (N, 1)
    
    # Launch kernel
    diag_matmul_kernel[grid](A, B, C, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)