import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tensor_matmul_kernel(
    A_ptr,  # Pointer to input tensor A (b, i, j, l)
    B_ptr,  # Pointer to input matrix B (l, k)
    C_ptr,  # Pointer to output tensor C (b, i, j, k)
    b,      # Batch size
    i,      # Height
    j,      # Width
    l,      # Inner dimension
    k,      # Output dimension
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute the block index for the output tensor
    pid = tl.program_id(0)
    
    # Calculate which batch, height, and width this block handles
    batch_idx = pid // (i * j)
    remaining = pid % (i * j)
    height_idx = remaining // j
    width_idx = remaining % j
    
    # Calculate pointers for this block
    a_ptr = A_ptr + batch_idx * i * j * l + height_idx * j * l + width_idx * l
    c_ptr = C_ptr + batch_idx * i * j * k + height_idx * j * k + width_idx * k
    
    # Loop over the output dimension (k)
    for k_idx in range(0, k, BLOCK_SIZE_K):
        # Initialize accumulator for this output element
        acc = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
        
        # Loop over the inner dimension (l) with appropriate blocking
        for l_idx in range(0, l, BLOCK_SIZE_L):
            # Load A slice (l,)
            a_mask = (l_idx + tl.arange(0, BLOCK_SIZE_L)) < l
            a_vals = tl.load(a_ptr + l_idx + tl.arange(0, BLOCK_SIZE_L), mask=a_mask, other=0.0)
            
            # Load B slice (l, k) - we need to load the appropriate column
            b_vals = tl.load(B_ptr + l_idx + tl.arange(0, BLOCK_SIZE_L)[:, None] * k + (k_idx + tl.arange(0, BLOCK_SIZE_K)[None, :]), 
                           mask=(l_idx + tl.arange(0, BLOCK_SIZE_L)[:, None] < l) & 
                                 (k_idx + tl.arange(0, BLOCK_SIZE_K)[None, :] < k), 
                           other=0.0)
            
            # Perform dot product
            acc += tl.sum(a_vals[:, None] * b_vals, axis=0)
        
        # Store results
        c_mask = (k_idx + tl.arange(0, BLOCK_SIZE_K)) < k
        tl.store(c_ptr + k_idx + tl.arange(0, BLOCK_SIZE_K), acc, mask=c_mask)

def triton_tensor_matmul(A, B):
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
    
    # Determine grid size
    grid_size = b * i * j
    
    # Launch kernel
    BLOCK_SIZE_L = 32
    BLOCK_SIZE_K = 32
    
    grid = lambda meta: (grid_size,)
    
    tensor_matmul_kernel[grid](A, B, C, b, i, j, l, k, BLOCK_SIZE_L, BLOCK_SIZE_K)
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_tensor_matmul(A, B)