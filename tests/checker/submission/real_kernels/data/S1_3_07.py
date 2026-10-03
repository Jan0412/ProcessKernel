import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Initialize pointers to the first memory blocks of A and B
    # A has shape (M, K) in this context (since we are doing A.T @ B.T, A.T is MxK)
    # B has shape (K, N) in this context (B.T is KxN)
    
    # Offsets for rows and columns
    row_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create pointers to A and B
    # A_ptr + row_offset[:, None] * stride_am + col_offset_k[None, :] * stride_ak
    # But we need to iterate over K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute offsets for K dimension
        k_offset = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Create pointers to A and B
        a_ptr = A_ptr + row_offset[:, None] * stride_am + k_offset[None, :] * stride_ak
        b_ptr = B_ptr + k_offset[:, None] * stride_bk + col_offset[None, :] * stride_bn
        
        # Create masks for A and B
        a_mask = (row_offset[:, None] < M) & (k_offset[None, :] < K)
        b_mask = (k_offset[:, None] < K) & (col_offset[None, :] < N)
        
        # Load A and B
        a = tl.load(a_ptr, mask=a_mask, other=0.0)
        b = tl.load(b_ptr, mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b)
        
    # Store result to C
    c_ptr = C_ptr + row_offset[:, None] * stride_cm + col_offset[None, :] * stride_cn
    c_mask = (row_offset[:, None] < M) & (col_offset[None, :] < N)
    tl.store(c_ptr, acc, mask=c_mask)

def triton_matmul_transpose(A, B):
    # A shape: (K, M)
    # B shape: (N, K)
    # Output shape: (M, N)
    
    M, K = A.shape # Wait, A is (K, M), so A.shape[1] is M, A.shape[0] is K
    # Let's rename to avoid confusion
    K_dim, M_dim = A.shape
    N_dim, K_dim_B = B.shape
    assert K_dim == K_dim_B, "Inner dimensions must match"
    
    M, N, K = M_dim, N_dim, K_dim
    
    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Strides for A.T (M, K)
    # A is (K, M). A.T[i, j] corresponds to A[j, i]
    # stride for row (M) is A.stride(0)
    # stride for col (K) is A.stride(1)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    
    # Strides for B.T (K, N)
    # B is (N, K). B.T[i, j] corresponds to B[j, i]
    # stride for row (K) is B.stride(1)
    # stride for col (N) is B.stride(0)
    stride_bk = B.stride(1)
    stride_bn = B.stride(0)
    
    # Strides for C (M, N) contiguous
    stride_cm = N
    stride_cn = 1
    
    # Grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul_transpose(A, B)