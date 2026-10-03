import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to first input tensor A (N, M, K)
    b_ptr,  # Pointer to second input tensor B (K, L)
    out_ptr,  # Pointer to output tensor (N, M, L)
    N, M, K, L,  # Dimensions
    stride_a_n, stride_a_m, stride_a_k,  # Strides for A
    stride_b_k, stride_b_l,  # Strides for B
    stride_out_n, stride_out_m, stride_out_l,  # Strides for output
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the block index for this program
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(L, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute the starting positions for this block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers to the input tensors
    a_ptrs = a_ptr + (offs_am[:, None] * stride_a_m + offs_k[None, :] * stride_a_k)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_b_k + offs_bn[None, :] * stride_b_l)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B tiles
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        
        # Perform matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_a_k
        b_ptrs += BLOCK_SIZE_K * stride_b_k
    
    # Compute output pointer
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (offs_am[:, None] * stride_out_m + offs_bn[None, :] * stride_out_l)
    
    # Store result
    tl.store(out_ptrs, accumulator, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < L))

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton kernel for 3D tensor-matrix multiplication.
    """
    assert a.is_cuda and b.is_cuda, "Both tensors must be on CUDA."
    assert a.dim() == 3 and b.dim() == 2, "A must be 3D and B must be 2D"
    assert a.shape[2] == b.shape[0], "Inner dimensions must match for matmul"
    
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    # Prepare output tensor
    N, M, K = a.shape
    L = b.shape[1]
    out = torch.empty((N, M, L), dtype=torch.float32, device=a.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Determine grid size
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(L, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, out,
        N, M, K, L,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
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
        Performs 3D tensor-matrix multiplication using custom Triton kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return triton_matmul(A, B)