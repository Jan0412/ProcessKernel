import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def einsum_bijl_lk_bijk_kernel(
    A_ptr,  # Pointer to input tensor A (b, i, j, l)
    B_ptr,  # Pointer to input matrix B (l, k)
    C_ptr,  # Pointer to output tensor C (b, i, j, k)
    b,      # Batch size
    i,      # First dimension
    j,      # Second dimension
    l,      # Third dimension (inner)
    k,      # Fourth dimension (output)
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the batch, i, j indices for this program
    batch_idx = tl.program_id(0)
    i_idx = tl.program_id(1)
    j_idx = tl.program_id(2)
    
    # Calculate the base offsets for this thread block
    batch_offset = batch_idx * i * j * l
    i_offset = i_idx * l
    j_offset = j_idx * l
    
    # Shared memory for matrix B
    B_shared = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_L, BLOCK_SIZE_K))
    
    # Loop over l dimension in chunks
    for l_start in range(0, l, BLOCK_SIZE_L):
        # Load B into shared memory
        l_offset = l_start + tl.arange(0, BLOCK_SIZE_L)
        k_offset = tl.arange(0, BLOCK_SIZE_K)
        
        # Ensure we don't go out of bounds
        l_mask = l_offset < l
        k_mask = k_offset < k
        
        # Load B[l_start:l_start+BLOCK_SIZE_L, :BLOCK_SIZE_K] into shared memory
        B_load_mask = tl.broadcast(l_mask[None, :], k_mask[:, None])
        B_vals = tl.load(B_ptr + l_offset[None, :] * k + k_offset[:, None], mask=B_load_mask, other=0.0)
        tl.store(B_shared, B_vals)
        
        # Compute partial results
        for k_idx in range(0, k, BLOCK_SIZE_K):
            # Calculate output offset
            out_offset = batch_idx * i * j * k + i_idx * j * k + j_idx * k + k_idx
            
            # Load A values
            a_offset = batch_offset + i_offset + j_offset + l_start
            a_vals = tl.load(A_ptr + a_offset + tl.arange(0, BLOCK_SIZE_L), mask=l_mask, other=0.0)
            
            # Perform matmul with shared B
            b_vals = tl.load(B_shared, mask=tl.broadcast(l_mask[None, :], k_mask[:, None]), other=0.0)
            
            # Compute dot product
            result = tl.sum(a_vals[:, None] * b_vals, axis=0)
            
            # Store result
            out_mask = k_idx + k_offset < k
            tl.store(C_ptr + out_offset + k_offset, result, mask=out_mask)

def triton_einsum_bijl_lk_bijk(A, B):
    """
    Custom Triton kernel for einsum operation: "bijl,lk->bijk"
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Tensors must be FP32."
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    b, i, j, l = A.shape
    l2, k = B.shape
    assert l == l2, "Inner dimensions must match"
    
    # Prepare output tensor
    C = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_L = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid = (
        b,      # batch dimension
        i,      # i dimension
        j       # j dimension
    )
    
    # Launch kernel
    einsum_bijl_lk_bijk_kernel[grid](
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
        return triton_einsum_bijl_lk_bijk(A, B)