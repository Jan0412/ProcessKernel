import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mv_kernel(
    A_ptr,
    B_ptr,
    Out_ptr,
    M, K,
    stride_Am, stride_Ak,
    stride_Bk,
    BLOCK_K: tl.constexpr,
):
    # Map program ID to row index m
    pid_m = tl.program_id(0)
    
    # Accumulator for the dot product
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Pointers to the start of the row in A and the vector B
    # A is (M, K), B is (K, 1)
    # We treat B as 1D for simplicity or just index into it.
    # A_ptr + pid_m * stride_Am
    
    A_row = A_ptr + pid_m * stride_Am
    B_vec = B_ptr
    
    # Loop over K dimension
    for off_k in range(0, K, BLOCK_K):
        # Create offsets for this block
        offsets_k = off_k + tl.arange(0, BLOCK_K)
        
        # Masking
        mask = offsets_k < K
        
        # Load A: shape (BLOCK_K,)
        # A is contiguous in last dim
        a_ptrs = A_row + offsets_k * stride_Ak
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        
        # Load B: shape (BLOCK_K,)
        # B is shape (K, 1), so stride is 1 if flattened or just offsets
        b_ptrs = B_vec + offsets_k * stride_Bk
        b = tl.load(b_ptrs, mask=mask, other=0.0)
        
        # Dot product
        acc += tl.sum(a * b)
        
    # Store result
    # Out is (M, 1)
    out_ptr = Out_ptr + pid_m * stride_Am # Assuming stride is consistent or just M*1
    # Actually Out shape is (M, 1), stride is 1 for the last dim.
    # Let's just use pid_m * 1 if it's contiguous.
    tl.store(Out_ptr + pid_m, acc)

def triton_mv(A, B):
    M, K = A.shape
    out = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    BLOCK_K = 1024 # Large block size for vector reduction
    
    grid = (M,)
    
    mv_kernel[grid](
        A, B, out,
        M, K,
        A.stride(0), A.stride(1), # strides
        B.stride(0),
        BLOCK_K
    )
    return out