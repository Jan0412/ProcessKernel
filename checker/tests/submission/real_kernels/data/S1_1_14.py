import torch
import triton
import triton.language as tl

@triton.jit
def matmul_tril_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a_offs = offs_am * stride_am + (k + tl.arange(0, BLOCK_K)) * stride_ak
        a_mask = (offs_am < M) & ((k + tl.arange(0, BLOCK_K)) < K)
        a = tl.load(A + a_offs, mask=a_mask, other=0.0)
        
        b_offs = (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_bn * stride_bn
        b_mask = ((k + tl.arange(0, BLOCK_K))[:, None] < K) & (offs_bn < N)
        b = tl.load(B + b_offs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)
        
    c_offs = offs_am * stride_cm + offs_bn * stride_cn
    c_mask = (offs_am < M) & (offs_bn < N)
    tl.store(C + c_offs, acc, mask=c_mask)

def triton_tril_matmul(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2 and M == N, "Shapes must match"
    
    C = torch.empty_like(A)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    
    matmul_tril_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
    )
    return C