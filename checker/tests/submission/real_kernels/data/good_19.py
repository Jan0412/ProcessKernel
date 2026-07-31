import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, i_size, j_size, l_size, k_size,
    stride_ab, stride_ai, stride_aj, stride_al,
    stride_bl, stride_bk,
    stride_cb, stride_ci, stride_cj, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the batch and spatial indices
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # Calculate the actual i and j indices
    i_idx = spatial_idx // j_size
    j_idx = spatial_idx % j_size
    
    # Create pointers for A and B
    a_ptr = A_ptr + batch_idx * stride_ab + i_idx * stride_ai + j_idx * stride_aj
    b_ptr = B_ptr + 0 * stride_bl + 0 * stride_bk
    
    # Create pointers for C
    c_ptr = C_ptr + batch_idx * stride_cb + i_idx * stride_ci + j_idx * stride_cj
    
    # Create a block of memory for the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, l_size, BLOCK_SIZE_K):
        # Load A block
        a_block = tl.load(
            tl.make_block_ptr(
                base=a_ptr,
                shape=(l_size, 1),
                strides=(stride_al, 1),
                offsets=(k, 0),
                block_shape=(BLOCK_SIZE_K, 1),
                order=(0, 1)
            )
        )
        
        # Load B block
        b_block = tl.load(
            tl.make_block_ptr(
                base=b_ptr,
                shape=(l_size, k_size),
                strides=(stride_bl, stride_bk),
                offsets=(k, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                order=(0, 1)
            )
        )
        
        # Perform matrix multiplication
        acc += tl.dot(a_block, b_block)
    
    # Write the result back to memory
    c_block = acc.to(tl.float32)
    tl.store(
        tl.make_block_ptr(
            base=c_ptr,
            shape=(1, k_size),
            strides=(stride_ck, 1),
            offsets=(0, 0),
            block_shape=(1, BLOCK_SIZE_N),
            order=(0, 1)
        ),
        c_block
    )

def triton_batched_matmul(A, B):
    """
    Performs batched matrix multiplication using Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 4 and B.dim() == 2, "A must be 4D and B must be 2D"
    assert A.shape[3] == B.shape[0], "Inner dimensions must match"
    
    # Prepare output tensor
    batch_size, i_size, j_size, l_size = A.shape
    k_size = B.shape[1]
    C = torch.empty(batch_size, i_size, j_size, k_size, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        (i_size * j_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    )
    
    # Launch kernel
    batched_matmul_kernel[grid](
        A, B, C,
        batch_size, i_size, j_size, l_size, k_size,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_batched_matmul(A, B)