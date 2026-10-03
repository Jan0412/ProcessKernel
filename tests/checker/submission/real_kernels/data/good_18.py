import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_4d_kernel(
    A_ptr,  # Pointer to 4D tensor A of shape (b, i, j, l)
    B_ptr,  # Pointer to matrix B of shape (l, k)
    C_ptr,  # Pointer to output tensor C of shape (b, i, j, k)
    b,      # Batch size
    i,      # First dimension
    j,      # Second dimension
    l,      # Third dimension (inner dimension)
    k,      # Fourth dimension
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the block IDs for batch, i, j dimensions
    batch_id = tl.program_id(0)
    i_id = tl.program_id(1)
    j_id = tl.program_id(2)
    
    # Calculate the starting indices for this block
    start_m = i_id * BLOCK_SIZE_M
    start_n = j_id * BLOCK_SIZE_N
    start_k = 0
    
    # Create output accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension in blocks
    for k_id in range(0, l, BLOCK_SIZE_K):
        # Load A block
        a_ptrs = A_ptr + batch_id * i * j * l + start_m * l + k_id
        a_mask = (start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < i) & \
                 (start_k + tl.arange(0, BLOCK_SIZE_K)[None, :] < l)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block
        b_ptrs = B_ptr + k_id * k + start_n
        b_mask = (start_k + tl.arange(0, BLOCK_SIZE_K)[:, None] < l) & \
                 (start_n + tl.arange(0, BLOCK_SIZE_N)[None, :] < k)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Perform matrix multiplication for this block
        acc += tl.dot(a, b)
    
    # Store the result
    c_ptrs = C_ptr + batch_id * i * j * k + start_m * k + start_n
    c_mask = (start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < i) & \
             (start_n + tl.arange(0, BLOCK_SIZE_N)[None, :] < k)
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_matmul_4d(A, B):
    """
    Performs 4D tensor-matrix multiplication using Triton kernel.
    
    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)
        
    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 4 and B.dim() == 2, "A must be 4D, B must be 2D"
    assert A.shape[3] == B.shape[0], "Inner dimensions must match"
    
    b, i, j, l = A.shape
    k = B.shape[1]
    
    # Prepare output tensor
    C = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid = (
        b,  # batch dimension
        (i + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,  # i dimension
        (j + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N   # j dimension
    )
    
    # Launch the kernel
    matmul_4d_kernel[grid](
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
        return triton_matmul_4d(A, B)