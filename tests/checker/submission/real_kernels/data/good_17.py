import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_4d_kernel(
    A_ptr,  # Pointer to 4D tensor A (b, i, j, l)
    B_ptr,  # Pointer to matrix B (l, k)
    C_ptr,  # Pointer to output tensor C (b, i, j, k)
    b,      # Batch size
    i,      # First spatial dimension
    j,      # Second spatial dimension
    l,      # Inner dimension
    k,      # Output dimension
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the block index for the batch dimension
    batch_idx = tl.program_id(0)
    # Get the block index for the spatial dimensions
    i_idx = tl.program_id(1)
    j_idx = tl.program_id(2)
    
    # Calculate the starting indices for this block
    start_m = i_idx * BLOCK_SIZE_M
    start_n = j_idx * BLOCK_SIZE_N
    start_k = 0
    
    # Create pointers for the output tensor
    c_ptr = C_ptr + batch_idx * i * j * k + start_m * j * k + start_n * k
    
    # Loop over the K dimension
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_idx in range(0, l, BLOCK_SIZE_K):
        # Load A block
        a_ptr = A_ptr + batch_idx * i * j * l + start_m * j * l + start_k
        a = tl.load(a_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] * j * l + 
                   tl.arange(0, BLOCK_SIZE_K)[None, :] * j * l + 
                   start_k, mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < i - start_m) &
                                 (tl.arange(0, BLOCK_SIZE_K)[None, :] < l - start_k))
        
        # Load B block
        b_ptr = B_ptr + start_k * k + tl.arange(0, BLOCK_SIZE_K)[:, None] * k + tl.arange(0, BLOCK_SIZE_N)[None, :]
        b = tl.load(b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * k + 
                   tl.arange(0, BLOCK_SIZE_N)[None, :], mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < l - start_k) &
                                                              (tl.arange(0, BLOCK_SIZE_N)[None, :] < k))
        
        # Perform matrix multiplication
        acc += tl.dot(a, b)
        
        # Update start_k for next iteration
        start_k += BLOCK_SIZE_K
    
    # Store the result
    tl.store(c_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] * k + 
             tl.arange(0, BLOCK_SIZE_N)[None, :], acc, 
             mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < i - start_m) &
                   (tl.arange(0, BLOCK_SIZE_N)[None, :] < k - start_n))

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
    assert A.dim() == 4 and B.dim() == 2, "A must be 4D and B must be 2D"
    assert A.shape[3] == B.shape[0], "Inner dimensions must match"
    
    b, i, j, l = A.shape
    k = B.shape[1]
    
    # Prepare output tensor
    C = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
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