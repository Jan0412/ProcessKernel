import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Block pointers
    # A is (M, K), B is (K, N) -> C is (M, N)
    # Because A and B are symmetric, M=N=K.
    
    # Offsets for rows and cols
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        # Load A tile
        # A is (M, K). Indices: [offs_m, k + tl.arange(0, BLOCK_K)]
        # Masking for boundaries
        mask_a = (offs_m[:, None] < M) & (k + tl.arange(0, BLOCK_K)[None, :] < K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_K)[None, :]) * stride_ak, mask=mask_a, other=0.0)
        
        # Load B tile
        # B is (K, N). Indices: [k + tl.arange(0, BLOCK_K), offs_n]
        mask_b = (k + tl.arange(0, BLOCK_K)[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_K)[:, None]) * stride_bk + offs_n[None, :] * stride_bn, mask=mask_b, other=0.0)
        
        # Matrix Multiply Accumulate
        acc += tl.dot(a, b)

    # Write back
    offs_m_res = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_res = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_c = (offs_m_res[:, None] < M) & (offs_n_res[None, :] < N)
    tl.store(c_ptr + offs_m_res[:, None] * stride_cm + offs_n_res[None, :] * stride_cn, acc, mask=mask_c)

def triton_matmul(A, B):
    # Check inputs
    assert A.shape == B.shape, "Matrices must be square and same shape for this specific symmetric model"
    assert A.shape[0] == A.shape[1], "Matrices must be square"
    
    M, N = A.shape
    K = B.shape[1] # Should be N
    
    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Launch config
    grid = (
        (M + 127) // 128, 
        (N + 127) // 128
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)