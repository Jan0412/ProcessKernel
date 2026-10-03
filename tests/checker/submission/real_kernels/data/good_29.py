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
    # Get the row and column indices for this thread
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Calculate the index in the flattened arrays
    a_idx = row
    b_idx = row * n_cols + col
    out_idx = row * n_cols + col
    
    # Load the diagonal element and the matrix element
    a_val = tl.load(a_ptr + a_idx)
    b_val = tl.load(b_ptr + b_idx)
    
    # Perform the multiplication
    out_val = a_val * b_val
    
    # Store the result
    tl.store(out_ptr + out_idx, out_val)

def triton_diag_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Triton-based implementation of diag(A) @ B
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
    
    # Create grid for 2D kernel (rows, cols)
    grid = (triton.cdiv(n_rows, BLOCK_SIZE), triton.cdiv(n_cols, BLOCK_SIZE))
    
    # Launch the kernel
    diag_matmul_kernel[grid](a, b, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)