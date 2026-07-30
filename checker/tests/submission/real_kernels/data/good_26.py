import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    a_ptr,  # Pointer to diagonal elements (1D)
    b_ptr,  # Pointer to matrix B (2D)
    c_ptr,  # Pointer to output matrix C (2D)
    n,      # Number of rows in B
    m,      # Number of columns in B
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate which row we're processing
    row = tl.program_id(0)
    
    # Each block processes one row
    if row >= n:
        return
        
    # Calculate offsets for this row
    a_offset = row
    b_row_offset = row * m
    c_row_offset = row * m
    
    # Process elements in blocks
    for col in range(0, m, BLOCK_SIZE):
        # Create mask for valid elements in this block
        mask = (col + tl.arange(0, BLOCK_SIZE)) < m
        
        # Load diagonal element
        a_val = tl.load(a_ptr + a_offset, mask=mask, other=0.0)
        
        # Load B row elements
        b_vals = tl.load(b_ptr + b_row_offset + col + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
        
        # Perform element-wise multiplication
        c_vals = a_val * b_vals
        
        # Store result
        tl.store(c_ptr + c_row_offset + col + tl.arange(0, BLOCK_SIZE), c_vals, mask=mask)

def triton_diag_matmul(A, B):
    """
    Optimized diagonal matrix multiplication using Triton kernel.
    Computes C = diag(A) @ B where A is 1D and B is 2D.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.shape[0] == B.shape[0], "Dimension mismatch between A and B"
    
    # Prepare output tensor
    C = torch.empty_like(B)
    
    # Get dimensions
    n, m = B.shape
    
    # Configure block size
    BLOCK_SIZE = 128
    
    # Launch kernel
    grid = (n, 1, 1)
    diag_matmul_kernel[grid](A, B, C, n, m, BLOCK_SIZE=BLOCK_SIZE)
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)