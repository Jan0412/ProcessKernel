import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first input (N, M, K)
    b_ptr,  # Pointer to second input (K, L)
    out_ptr,  # Pointer to output (N, M, L)
    N, M, K, L,  # Dimensions
    stride_a_n, stride_a_m, stride_a_k,  # Strides for A
    stride_b_k, stride_b_l,  # Strides for B
    stride_out_n, stride_out_m, stride_out_l,  # Strides for output
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the block IDs
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)
    
    # Compute the starting indices for this block
    start_m = block_id_m * BLOCK_SIZE_M
    start_n = block_id_n * BLOCK_SIZE_N
    
    # Create the output tile
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A tile
        a_tile = tl.load(
            a_ptr + start_m * stride_a_n + k * stride_a_k,
            mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < (N - start_m)) &
                  (tl.arange(0, BLOCK_SIZE_K)[None, :] < (K - k)),
            other=0.0
        )
        
        # Load B tile
        b_tile = tl.load(
            b_ptr + k * stride_b_k,
            mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < (K - k)) &
                  (tl.arange(0, BLOCK_SIZE_N)[None, :] < (L)),
            other=0.0
        )
        
        # Perform matrix multiplication for this tile
        accumulator = tl.dot(a_tile, b_tile, accumulator)
    
    # Write the output tile
    out_tile = accumulator
    tl.store(
        out_ptr + start_m * stride_out_n + start_n * stride_out_l,
        out_tile,
        mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < (N - start_m)) &
              (tl.arange(0, BLOCK_SIZE_N)[None, :] < (L))
    )

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Triton-based 3D tensor-matrix multiplication.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    assert a.dim() == 3 and b.dim() == 2, "Input dimensions incorrect"
    assert a.shape[2] == b.shape[0], "Matrix dimensions incompatible"
    
    # Prepare tensors for contiguous memory access
    a = a.contiguous()
    b = b.contiguous()
    
    # Extract dimensions
    N, M, K = a.shape
    L = b.shape[1]
    
    # Create output tensor
    out = torch.empty(N, M, L, dtype=torch.float32, device=a.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (L + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Launch kernel
    matmul_kernel[grid](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        out_ptr=out.data_ptr(),
        N=N, M=M, K=K, L=L,
        stride_a_n=a.stride(0), stride_a_m=a.stride(1), stride_a_k=a.stride(2),
        stride_b_k=b.stride(0), stride_b_l=b.stride(1),
        stride_out_n=out.stride(0), stride_out_m=out.stride(1), stride_out_l=out.stride(2),
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
        Performs 3D tensor-matrix multiplication using Triton kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return triton_matmul(A, B)