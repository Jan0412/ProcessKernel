import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Pointers to A and B
    a_ptrs = A + off_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptrs = B + tl.arange(0, BLOCK_K)[:, None] * stride_bk + off_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        # Load A and B blocks
        a = tl.load(a_ptrs, mask=tl.arange(0, BLOCK_K)[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=tl.arange(0, BLOCK_K)[:, None] < K - k, other=0.0)
        # Accumulate
        acc += tl.dot(a, b)
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write to C
    c_ptrs = C + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(off_m[:, None] < M) & (off_n[None, :] < N))

def triton_matmul(A, B):
    # A: (M, K), B: (N, K) -> C: (M, N) = A @ B.T
    M, K = A.shape
    N, _ = B.shape
    
    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Strides
    stride_am, stride_ak = A.stride()
    stride_bn, stride_bk = B.stride()
    stride_cm, stride_cn = C.stride()
    
    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 32
    
    # Grid
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)