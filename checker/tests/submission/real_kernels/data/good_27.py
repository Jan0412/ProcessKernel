import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    a_ptr,  # Pointer to diagonal elements (1D)
    b_ptr,  # Pointer to matrix B (2D)
    out_ptr,  # Pointer to output matrix C (2D)
    n_rows,  # Number of rows in B and C
    n_cols,  # Number of columns in B and C
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Each program handles one row
    if row_idx >= n_rows:
        return
    
    # Calculate the starting offset for this row in B and C
    row_offset_b = row_idx * n_cols
    row_offset_out = row_idx * n_cols
    
    # Load the diagonal element for this row
    a_val = tl.load(a_ptr + row_idx)
    
    # Process columns in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Calculate column offsets
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load B[row, :] values
        b_vals = tl.load(b_ptr + row_offset_b + col_offsets, mask=mask, other=0.0)
        
        # Perform element-wise multiplication: C[row, :] = A[row] * B[row, :]
        out_vals = a_val * b_vals
        
        # Store results
        tl.store(out_ptr + row_offset_out + col_offsets, out_vals, mask=mask)

def triton_diag_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Optimized implementation of diag(A) @ B using Triton kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    assert a.dim() == 1, "A must be 1D"
    assert b.dim() == 2, "B must be 2D"
    assert a.size(0) == b.size(0), "Dimension mismatch between A and B"
    
    # Prepare output tensor
    out = torch.empty_like(b)
    
    # Get dimensions
    n_rows, n_cols = b.shape
    
    # Configure block size
    BLOCK_SIZE = 128
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    diag_matmul_kernel[grid](a, b, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)