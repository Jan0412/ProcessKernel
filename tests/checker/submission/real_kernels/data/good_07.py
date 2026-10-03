import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_3d_kernel(
    A_ptr, B_ptr, out_ptr,
    N, M, K, L,
    stride_a_n, stride_a_m, stride_a_k,
    stride_b_k, stride_b_l,
    stride_out_n, stride_out_m, stride_out_l,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the block index
    block_id = tl.program_id(0)
    
    # Compute the block indices
    block_m = block_id // (L // BLOCK_SIZE_N)
    block_n = block_id % (L // BLOCK_SIZE_N)
    
    # Compute the starting indices for this block
    start_m = block_m * BLOCK_SIZE_M
    start_n = block_n * BLOCK_SIZE_N
    
    # Create the output accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the A block
        a = tl.load(
            A_ptr + 
            start_m * stride_a_n + 
            tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_a_m + 
            k * stride_a_k + 
            tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_a_k,
            mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < (N - start_m)) &
                 (tl.arange(0, BLOCK_SIZE_K)[None, :] < (K - k)),
            other=0.0
        )
        
        # Load the B block
        b = tl.load(
            B_ptr + 
            k * stride_b_k + 
            start_n * stride_b_l + 
            tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_b_k + 
            tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_b_l,
            mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < (K - k)) &
                 (tl.arange(0, BLOCK_SIZE_N)[None, :] < (L - start_n)),
            other=0.0
        )
        
        # Perform the matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Store the result
    out = accumulator
    
    # Write the output
    tl.store(
        out_ptr + 
        start_m * stride_out_n + 
        start_n * stride_out_l + 
        tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_out_m + 
        tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_out_l,
        out,
        mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < (N - start_m)) &
             (tl.arange(0, BLOCK_SIZE_N)[None, :] < (L - start_n))
    )

def triton_matmul_3d(A, B):
    """
    Performs 3D tensor-matrix multiplication using Triton kernel.
    
    Args:
        A (torch.Tensor): Input 3D tensor of shape (N, M, K).
        B (torch.Tensor): Input matrix of shape (K, L).
        
    Returns:
        torch.Tensor: Output tensor of shape (N, M, L).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dim() == 3 and B.dim() == 2, "A must be 3D and B must be 2D"
    assert A.shape[2] == B.shape[0], "Inner dimensions must match"
    
    # Ensure tensors are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    N, M, K = A.shape
    L = B.shape[1]
    out = torch.empty(N, M, L, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid = lambda meta: (
        (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * (L + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )
    
    # Launch the kernel
    matmul_3d_kernel[grid](
        A, B, out,
        N, M, K, L,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using optimized Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using optimized Triton kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return triton_matmul_3d(A, B)