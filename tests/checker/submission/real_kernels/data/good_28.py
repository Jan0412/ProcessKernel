import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def diag_matmul_kernel(
    a_ptr,  # Pointer to diagonal elements (1D)
    b_ptr,  # Pointer to matrix B (2D)
    out_ptr,  # Pointer to output matrix (2D)
    n_rows,  # Number of rows in B
    n_cols,  # Number of columns in B
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Each program processes one row
    if row_idx >= n_rows:
        return
    
    # Calculate offsets for this row
    a_offset = row_idx
    b_offset = row_idx * n_cols
    out_offset = row_idx * n_cols
    
    # Process elements in blocks
    for col_idx in range(0, n_cols, BLOCK_SIZE):
        # Create offsets for this block
        offsets = col_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        a_val = tl.load(a_ptr + a_offset, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + b_offset + offsets, mask=mask, other=0.0)
        
        # Perform element-wise multiplication
        out_vals = a_val * b_vals
        
        # Store results
        tl.store(out_ptr + out_offset + offsets, out_vals, mask=mask)

def triton_diag_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Optimized diagonal matrix multiplication using Triton kernel.
    Computes diag(a) @ b efficiently without creating the full diagonal matrix.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    assert a.dim() == 1, "A must be 1D"
    assert b.dim() == 2, "B must be 2D"
    assert a.size(0) == b.size(0), "Dimension mismatch between A and B"
    
    a = a.contiguous()
    b = b.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(b)
    
    # Get dimensions
    n_rows, n_cols = b.shape
    BLOCK_SIZE = 128  # Tunable parameter
    
    # Determine grid size
    grid = (triton.cdiv(n_rows, 1),)
    
    # Launch kernel
    diag_matmul_kernel[grid](a, b, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
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
        C = diag(A) * B

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)