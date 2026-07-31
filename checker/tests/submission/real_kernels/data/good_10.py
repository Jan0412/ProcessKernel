import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_3d_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_an, stride_ak,
    stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map program ID to the block of the output matrix
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Which block of the output matrix to compute
    group_id = pid // GROUP_M
    first_m = group_id * GROUP_M * BLOCK_M
    remainder = pid % GROUP_M
    m_start = first_m + remainder * BLOCK_M
    n_start = (pid % GROUP_M) * BLOCK_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tiles
        a = tl.load(tl.make_block_ptr(
            a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(m_start, k), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
        ))
        b = tl.load(tl.make_block_ptr(
            b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(k, n_start), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0)
        ))
        
        # Compute partial matrix multiplication
        acc += tl.dot(a, b)
    
    # Apply activation if needed
    if ACTIVATION == "relu":
        acc = tl.where(acc > 0, acc, 0.0)
    
    # Store result
    out = tl.make_block_ptr(
        out_ptr, shape=(M, N), strides=(stride_out_m, stride_out_n),
        offsets=(m_start, n_start), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    tl.store(out, acc, boundary_check=(0, 1))

def triton_matmul_3d(a, b):
    """
    Performs 3D tensor-matrix multiplication using Triton kernel.
    
    Args:
        a (torch.Tensor): Input 3D tensor of shape (N, M, K)
        b (torch.Tensor): Input matrix of shape (K, L)
        
    Returns:
        torch.Tensor: Output tensor of shape (N, M, L)
    """
    # Ensure inputs are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous()
    
    # Get dimensions
    N, M, K = a.shape
    L = b.shape[1]
    
    # Create output tensor
    out = torch.empty((N, M, L), device=a.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8
    
    # Calculate grid size
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(L, meta["BLOCK_N"]),
    )
    
    # Launch kernel
    matmul_3d_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        ACTIVATION="",
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
        return triton_matmul_3d(A, B)