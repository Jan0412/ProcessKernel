import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Each program handles one row
    if row_idx >= n_rows:
        return
    
    # Calculate offsets for this row
    a_offset = row_idx
    b_offset = row_idx * n_cols
    
    # Load the scalar value from A
    a_val = tl.load(a_ptr + a_offset)
    
    # Process columns in blocks
    for col_idx in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual column index
        actual_col = col_idx + tl.arange(0, BLOCK_SIZE)
        mask = actual_col < n_cols
        
        # Load B values for this row and column range
        b_vals = tl.load(b_ptr + b_offset + actual_col, mask=mask, other=0.0)
        
        # Perform element-wise multiplication
        out_vals = a_val * b_vals
        
        # Store results
        tl.store(out_ptr + b_offset + actual_col, out_vals, mask=mask)

def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized implementation of diag(A) @ B using Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 1, "A must be 1D"
    assert B.dim() == 2, "B must be 2D"
    assert A.size(0) == B.size(0), "Dimension mismatch between A and B"
    
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(B)
    
    # Get dimensions
    n_rows, n_cols = B.shape
    BLOCK_SIZE = 128
    
    # Calculate grid
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    diag_matmul_kernel[grid](A, B, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for diagonal matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using optimized Triton kernel.
        
        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)