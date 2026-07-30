import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        # Offsets for K dimension
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Load A and B
        a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, 
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, 
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
        # Multiply and accumulate
        acc += tl.dot(a, b)

    # Store result
    C_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_matmul(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']), 
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=128, BLOCK_N=256, BLOCK_K=64
    )
    return C