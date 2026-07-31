import torch
import triton
import triton.language as tl

@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    B, M, N,
    stride_xb, stride_xm, stride_xn,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one (b, n) position
    pid = tl.program_id(0)
    
    # Calculate b and n indices
    n = pid % N
    b = pid // N
    
    # Base offset for x
    offset_x = b * stride_xb + n * stride_xn
    
    # Initialize min_val and min_idx
    min_val = tl.full((1,), float('inf'), dtype=tl.float32)
    min_idx = tl.full((1,), -1, dtype=tl.int32)
    
    # Iterate over M dimension in blocks
    for m_start in range(0, M, BLOCK_SIZE):
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE)
        mask = m_offsets < M
        
        # Load values
        x_vals = tl.load(x_ptr + offset_x + m_offsets * stride_xm, mask=mask, other=float('inf'))
        
        # Find local min
        local_min_val = tl.min(x_vals)
        local_min_idx = tl.argmin(x_vals) + m_start
        
        # Update global min
        # If local min is smaller than current global min
        if local_min_val < min_val:
            min_val = local_min_val
            min_idx = local_min_idx
            
        # Optimization: If we found -inf, we can't get lower, but argmin usually handles ties by first index.
        # Standard reduction logic continues.
        
    # Store result
    tl.store(out_ptr + pid, min_idx)

def triton_argmin(x: torch.Tensor, dim: int = 1):
    assert x.is_cuda
    x = x.contiguous()
    B, M, N = x.shape
    assert dim == 1, "Only dim=1 supported in this optimized kernel"
    
    out = torch.empty((B, N), dtype=torch.int64, device=x.device)
    
    grid = (B * N,)
    
    BLOCK_SIZE = 128 # Or 256, 512. M=4096. 128 is safe.
    
    argmin_kernel[grid](
        x, out,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out