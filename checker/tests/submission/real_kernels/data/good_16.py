import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_4d_kernel(
    A_ptr,  # Pointer to 4D tensor A (b, i, j, l)
    B_ptr,  # Pointer to matrix B (l, k)
    C_ptr,  # Pointer to output tensor C (b, i, j, k)
    b, i, j, l, k,  # Dimensions
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the block indices
    b_idx = tl.program_id(0)
    i_idx = tl.program_id(1)
    j_idx = tl.program_id(2)
    
    # Compute the starting positions for this block
    offs_b = b_idx
    offs_i = i_idx
    offs_j = j_idx
    
    # Create pointers to the relevant slices of A and C
    A_base = A_ptr + offs_b * i * j * l + offs_i * j * l + offs_j * l
    C_base = C_ptr + offs_b * i * j * k + offs_i * j * k + offs_j * k
    
    # Loop over k dimension in blocks
    for k_start in range(0, k, BLOCK_SIZE_K):
        # Initialize accumulator
        acc = tl.zeros((1, 1), dtype=tl.float32)
        
        # Loop over l dimension in blocks
        for l_start in range(0, l, BLOCK_SIZE_L):
            # Load A slice
            l_offsets = l_start + tl.arange(0, BLOCK_SIZE_L)
            mask_l = l_offsets < l
            
            A_vals = tl.load(A_base + l_offsets, mask=mask_l, other=0.0)
            
            # Load B slice
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
            mask_k = k_offsets < k
            
            B_vals = tl.load(B_ptr + l_offsets[:, None] * k + k_offsets[None, :], mask=mask_k[None, :], other=0.0)
            
            # Compute partial dot product
            acc += tl.sum(A_vals[:, None] * B_vals, axis=0)
        
        # Store the result
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < k
        tl.store(C_base + k_offsets, acc, mask=mask_k)

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
    BLOCK_SIZE_L = 32
    BLOCK_SIZE_K = 32
    
    # Create grid
    grid = (b, i, j)
    
    # Launch kernel
    matmul_4d_kernel[grid](
        A, B, C,
        b, i, j, l, k,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul_4d(A, B)