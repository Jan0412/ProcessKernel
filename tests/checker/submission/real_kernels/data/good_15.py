import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    stride_ab, stride_ai, stride_aj, stride_al,
    stride_bl, stride_bk,
    stride_ob, stride_oi, stride_oj, stride_ok,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Identify the block index
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers for the A and B matrices
    a_ptrs = a_ptr + (offs_am[:, None] * stride_ab + offs_k[None, :] * stride_al)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bl + offs_bn[None, :] * stride_bk)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B tiles
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
        
        # Compute accumulator
        accumulator += tl.dot(a, b)
        
        # Update pointers
        a_ptrs += BLOCK_SIZE_K * stride_al
        b_ptrs += BLOCK_SIZE_K * stride_bl
    
    # Compute output pointer
    offs_o = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_oj = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (offs_o[:, None] * stride_ob + offs_oj[None, :] * stride_ok)
    
    # Store result
    tl.store(out_ptrs, accumulator, mask=(offs_o[:, None] < M) & (offs_oj[None, :] < N))

def triton_matmul_4d(a, b):
    """
    Performs 4D tensor-matrix multiplication: C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    assert a.dim() == 4 and b.dim() == 2, "Input dimensions incorrect"
    assert a.shape[3] == b.shape[0], "Matrix dimensions incompatible"
    
    # Reshape tensors for efficient computation
    b, i, j, l = a.shape
    l2, k = b.shape
    assert l == l2, "Matrix dimensions incompatible"
    
    # Prepare output tensor
    out = torch.empty(b, i, j, k, dtype=torch.float32, device=a.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Determine grid size
    grid = lambda meta: (
        triton.cdiv(i * j, meta["BLOCK_SIZE_M"]) * triton.cdiv(k, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, out,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        i * j, k, l,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul_4d(A, B)