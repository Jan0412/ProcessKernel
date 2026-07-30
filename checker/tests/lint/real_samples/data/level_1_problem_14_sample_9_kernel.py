import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triu_matmul_kernel(
    A, B, out,
    N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and Column offsets
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for N x N bounds
    row_mask = row_idx < N
    col_mask = col_idx < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Upper triangular check: we only compute if col_idx >= row_idx
    # We create a mask where col >= row. 
    # Note: row_idx is (BLOCK_M,), col_idx is (BLOCK_N,)
    # Broadcasting in Triton works element-wise in the loop or via explicit broadcasting
    upper_tri_mask = (col_idx[None, :] >= row_idx[:, None])
    
    # Combine masks
    valid_mask = row_mask[:, None] & col_mask[None, :] & upper_tri_mask

    # Loop over K
    for k in range(0, N, BLOCK_K):
        # Load A: shape (BLOCK_M, BLOCK_K)
        a_idx = row_idx[:, None] + tl.arange(0, BLOCK_K)[None, :]
        a_mask = row_mask[:, None] & (a_idx < N)
        a = tl.load(A + a_idx, mask=a_mask, other=0.0)
        
        # Load B: shape (BLOCK_K, BLOCK_N)
        b_idx = tl.arange(0, BLOCK_K)[:, None] + col_idx[None, :]
        b_mask = (b_idx < N) & col_mask[None, :]
        b = tl.load(B + b_idx, mask=b_mask, other=0.0)

        # Matrix multiply and accumulate
        acc += tl.dot(a, b)

    # Store result
    # If not in upper triangle, we write 0 (which is already in acc if we didn't compute, 
    # but we computed everything. We need to zero out the lower triangle).
    # Actually, simpler: just mask the store.
    tl.store(out + row_idx[:, None] * N + col_idx[None, :], acc, mask=valid_mask)

def triu_matmul(A, B):
    N = A.shape[0]
    out = torch.zeros((N, N), device=A.device, dtype=torch.float32)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = (
        triton.cdiv(N, BLOCK_M), 
        triton.cdiv(N, BLOCK_N)
    )
    
    triu_matmul_kernel[grid](A, B, out, N, BLOCK_M, BLOCK_N, BLOCK_K)
    return out