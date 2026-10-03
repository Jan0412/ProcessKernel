import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triangular_matmul_kernel(
    A, B, C,
    N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, N, BLOCK_K):
        # Offsets for A: (BLOCK_M x BLOCK_K)
        a_row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        a_col = k + tl.arange(0, BLOCK_K)[None, :]
        
        # Offsets for B: (BLOCK_K x BLOCK_N)
        b_row = k + tl.arange(0, BLOCK_K)[:, None]
        b_col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        # Masks for A (upper triangular: row <= col) and boundary
        a_mask = (a_row <= a_col) & (a_row < N) & (a_col < N)
        a = tl.load(A + a_row * N + a_col, mask=a_mask, other=0.0)

        # Masks for B (upper triangular: row <= col) and boundary
        b_mask = (b_row <= b_col) & (b_row < N) & (b_col < N)
        b = tl.load(B + b_row * N + b_col, mask=b_mask, other=0.0)

        # Dot product
        acc = tl.dot(a, b)

    # Store result C (upper triangular: row <= col)
    c_row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    c_col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    c_mask = (c_row <= c_col) & (c_row < N) & (c_col < N)
    tl.store(C + c_row * N + c_col, acc, mask=c_mask)

def triton_upper_triangular_matmul(A, B):
    N = A.shape[0]
    assert A.shape == (N, N) and B.shape == (N, N)
    C = torch.empty((N, N), dtype=A.dtype, device=A.device)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))
    triangular_matmul_kernel[grid](A, B, C, N, BLOCK_M, BLOCK_N, BLOCK_K)
    return C