import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N, C, H, W,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate global offset for this block
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = N * C * H * W
    mask = offsets < total_elements
    
    # Calculate channel index for each element in the block
    # offset // (H * W) gives n * C + c
    # % C gives c
    channel = (offsets // (H * W)) % C
    
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load mean and var using channel index
    # We use tl.gather if necessary, or just simple indexing
    mean = tl.load(mean_ptr + channel, mask=mask)
    var = tl.load(var_ptr + channel, mask=mask)
    
    # Load gamma and beta
    gamma = tl.load(gamma_ptr + channel, mask=mask)
    beta = tl.load(beta_ptr + channel, mask=mask)
    
    # Compute normalized value
    # out = gamma * (x - mean) / sqrt(var + eps) + beta
    out = gamma * (x - mean) / tl.sqrt(var + eps) + beta
    
    # Store out
    tl.store(out_ptr + offsets, out, mask=mask)

def batch_norm_forward_triton(x, running_mean, running_var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 256
    total_elements = N * C * H * W
    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)
    
    batch_norm_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        N, C, H, W, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out