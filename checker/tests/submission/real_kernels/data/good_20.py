import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to input tensor A (b, i, j, l)
    b_ptr,  # Pointer to input matrix B (l, k)
    c_ptr,  # Pointer to output tensor C (b, i, j, k)
    b_size, i_size, j_size, l_size, k_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the program ID for batch, row, and column
    batch_id = tl.program_id(0)
    row_id = tl.program_id(1)
    col_id = tl.program_id(2)
    
    # Calculate the starting indices for this program
    start_m = row_id * BLOCK_SIZE_M
    start_n = col_id * BLOCK_SIZE_N
    start_k = 0
    
    # Create accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension in blocks
    for k in range(0, l_size, BLOCK_SIZE_K):
        # Load A block
        a_block = tl.load(
            a_ptr + 
            batch_id * i_size * j_size * l_size +
            start_m * j_size * l_size +
            tl.arange(0, BLOCK_SIZE_M)[:, None] * j_size * l_size +
            start_k + tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        
        # Load B block
        b_block = tl.load(
            b_ptr + 
            start_k * k_size +
            tl.arange(0, BLOCK_SIZE_K)[:, None] * k_size +
            tl.arange(0, BLOCK_SIZE_N)[None, :]
        )
        
        # Perform matrix multiplication for this block
        acc += tl.dot(a_block, b_block)
    
    # Store the result
    tl.store(
        c_ptr + 
        batch_id * i_size * j_size * k_size +
        start_m * j_size * k_size +
        tl.arange(0, BLOCK_SIZE_M)[:, None] * j_size * k_size +
        start_n + tl.arange(0, BLOCK_SIZE_N)[None, :],
        acc
    )

def triton_matmul(A, B):
    """
    Custom Triton implementation of 4D tensor-matrix multiplication
    """
    # Ensure inputs are contiguous and on GPU
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    b, i, j, l = A.shape
    l2, k = B.shape
    
    # Validate dimensions
    assert l == l2, f"Dimension mismatch: A has {l} in last dim, B has {l2}"
    
    # Create output tensor
    C = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid = (
        b,  # batch dimension
        triton.cdiv(i, BLOCK_SIZE_M),  # row dimension
        triton.cdiv(k, BLOCK_SIZE_N)   # column dimension
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        b, i, j, l, k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)