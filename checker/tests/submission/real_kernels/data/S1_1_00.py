import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    stride_an, stride_am, stride_ak,
    stride_bk, stride_bl,
    stride_cn, stride_cm, stride_cl,
    M, K, L, N,
    BLOCK_M: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1. Program IDs
    pid_m = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    # 2. Offsets for M and L
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    off_k = tl.arange(0, BLOCK_K)
    
    # 3. Loop over N
    # We can't easily loop over N inside the kernel if we want to use the accumulator
    # efficiently for the tiled matmul, because the accumulator needs to be reset for each N.
    # Wait, we can just define `acc = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)` inside the N loop.
    
    # Actually, it's better to launch a 3D grid (N, M, L) if N is small?
    # N=16 is small. Launching 16 * ceil(1024/128) * ceil(768/128) = 16 * 8 * 6 = 768 blocks.
    # This is very small for GPU. We might want to keep N in the loop to ensure occupancy or just launch a 2D grid and loop N.
    # Let's stick to 2D grid (M, L) and loop N.
    
    # Initialize output pointer base for the current (m, l) block
    # We will write to C[n, m, l]
    
    for n in range(N):
        acc = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)
        
        # Pointers to A and B
        # A shape (N, M, K). B shape (K, L)
        # For a given n, A_ptr + n * stride_an
        
        # Loop over K
        for k in range(0, K, BLOCK_K):
            # Load A
            a_mask = off_m[:, None] < M
            a = tl.load(A + n * stride_an + off_m[:, None] * stride_am + (off_k[None, :] + k) * stride_ak, mask=a_mask, other=0.0)
            
            # Load B
            b_mask = (off_k[:, None] + k) < K
            b = tl.load(B + (off_k[:, None] + k) * stride_bk + off_l[None, :] * stride_bl, mask=b_mask, other=0.0)
            
            acc = tl.dot(a, b, acc)
        
        # Store C
        c_mask = (off_m[:, None] < M) & (off_l[None, :] < L)
        tl.store(C + n * stride_cn + off_m[:, None] * stride_cm + off_l[None, :] * stride_cl, acc, mask=c_mask)

def triton_matmul(A, B):
    # ... setup ...
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(L, BLOCK_L))
    matmul_kernel[grid](...)
    return C