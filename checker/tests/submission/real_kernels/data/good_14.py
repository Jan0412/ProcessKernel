import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to input tensor A (b, i, j, l)
    b_ptr,  # Pointer to input tensor B (l, k)
    c_ptr,  # Pointer to output tensor C (b, i, j, k)
    b_size,  # Batch size
    i_size,  # First dimension of A and C
    j_size,  # Second dimension of A and C
    l_size,  # Third dimension of A and second dimension of B
    k_size,  # Fourth dimension of C and second dimension of B
    stride_a_b, stride_a_i, stride_a_j, stride_a_l,  # Strides for A
    stride_b_l, stride_b_k,  # Strides for B
    stride_c_b, stride_c_i, stride_c_j, stride_c_k,  # Strides for C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    # Get the row and column index for this program
    row_idx = tl.program_id(1)
    col_idx = tl.program_id(2)
    
    # Create a block of indices for the M and N dimensions
    m_offsets = row_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = col_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for valid indices
    m_mask = m_offsets < i_size
    n_mask = n_offsets < k_size
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, l_size, BLOCK_SIZE_K):
        # Create offsets for K dimension
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < l_size
        
        # Load A values
        a_offsets = batch_idx * stride_a_b + m_offsets[:, None] * stride_a_i + \
                    tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_a_l
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load B values
        b_offsets = k_offsets[None, :] * stride_b_l + n_offsets[:, None] * stride_b_k
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Perform matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Write result to output tensor
    c_offsets = batch_idx * stride_c_b + m_offsets[:, None] * stride_c_i + \
                n_offsets[None, :] * stride_c_k
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)

def triton_matmul(A, B):
    """
    Performs 4D tensor-matrix multiplication using Triton kernel.
    
    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)
    
    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
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
    
    # Create output tensor
    C = torch.empty(b, i, j, k, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid = (
        b,  # Batch dimension
        triton.cdiv(i, BLOCK_SIZE_M),  # Row dimension
        triton.cdiv(k, BLOCK_SIZE_N)   # Column dimension
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        b, i, j, l, k,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)