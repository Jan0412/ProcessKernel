import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    b, i, j, l, k,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension
    stride_ab, stride_ai, stride_aj, stride_al,
    stride_bl, stride_bk,
    stride_cb, stride_ci, stride_cj, stride_ck,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    # Map program id to the block of C it should compute
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(i, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(k, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute the starting indices for this block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers for the first blocks of A and B
    a_ptrs = a_ptr + (
        tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_ai +
        offs_k[None, :] * stride_al
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bl +
        tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_bk
    )

    # Initialize accumulator with the same type as the input matrices
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)
    
    # Loop over K dimension
    for k in range(0, l, BLOCK_SIZE_K):
        # Load A and B tiles
        a = tl.load(a_ptrs, mask=offs_am[:, None] < i and offs_k[None, :] < l - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < l - k and offs_bn[None, :] < k, other=0.0)
        
        # Compute the matrix multiplication
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_al
        b_ptrs += BLOCK_SIZE_K * stride_bl

    # Compute the output pointer
    c_ptrs = c_ptr + (
        pid_m * BLOCK_SIZE_M * stride_ci +
        pid_n * BLOCK_SIZE_N * stride_ck
    )
    
    # Store the result
    tl.store(c_ptrs, acc, mask=offs_am[:, None] < i and offs_bn[None, :] < k)


def triton_matmul(A, B):
    """
    Performs 4D tensor-matrix multiplication using Triton kernel.
    A: (b, i, j, l) tensor
    B: (l, k) matrix
    Returns: (b, i, j, k) tensor
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Tensors must be FP32"
    
    b, i, j, l = A.shape
    l2, k = B.shape
    assert l == l2, "Inner dimensions must match"
    
    # Create output tensor
    C = torch.empty((b, i, j, k), device=A.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Determine the number of blocks needed
    grid = lambda meta: (
        triton.cdiv(i, meta["BLOCK_SIZE_M"]) * triton.cdiv(k, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        b, i, j, l, k,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACC_TYPE=tl.float32,
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)