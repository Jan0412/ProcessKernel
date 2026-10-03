import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_reduce_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    n, c, h, w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute mean and variance per (N, C) instance.
    Each program handles one (N, C) pair.
    """
    idx = tl.program_id(0)  # idx in [0, N*C)
    n_idx = idx // c
    c_idx = idx % c
    
    # Base pointer for this (N, C) slice
    base_ptr = x_ptr + n_idx * stride_xn + c_idx * stride_xc
    
    # Initialize accumulators
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over all H*W elements in tiles
    num_elements = h * w
    for start in range(0, num_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        
        # Linear index to (h, w) coordinates
        h_idx = offsets // w
        w_idx = offsets % w
        
        # Compute pointer
        ptr = base_ptr + h_idx * stride_xh + w_idx * stride_xw
        x = tl.load(ptr, mask=mask, other=0.0)
        
        sum_val += x
        sum_sq += x * x
    
    # Reduce across BLOCK_SIZE elements
    # Use a simple loop reduction
    for i in range(1, BLOCK_SIZE):
        sum_val[0] += sum_val[i]
        sum_sq[0] += sum_sq[i]
    
    # Compute mean and variance
    mean = sum_val[0] / num_elements
    var = sum_sq[0] / num_elements - mean * mean
    
    # Store results
    tl.store(mean_ptr + idx, mean)
    tl.store(var_ptr + idx, var)


@triton.jit
def instance_norm_apply_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n, c, h, w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply normalization: out = (x - mean) / sqrt(var + eps) * weight + bias
    Each program handles a block of elements.
    """
    pid = tl.program_id(0)
    
    # Flatten the output space
    flat_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = n * c * h * w
    mask = flat_idx < total_elements
    
    # Decode flat index to (n, c, h, w)
    w_idx = flat_idx % w
    flat_idx_hwc = flat_idx // w
    h_idx = flat_idx_hwc % h
    flat_idx_nc = flat_idx_hwc // h
    c_idx = flat_idx_nc % c
    n_idx = flat_idx_nc // c
    
    # Compute (N, C) index for mean/var/weight/bias
    nc_idx = n_idx * c + c_idx
    
    # Load x
    x_ptr_offset = n_idx * stride_xn + c_idx * stride_xc + h_idx * stride_xh + w_idx * stride_xw
    x = tl.load(x_ptr + x_ptr_offset, mask=mask, other=0.0)
    
    # Load mean, var, weight, bias
    mean = tl.load(mean_ptr + nc_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + nc_idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + c_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    
    # Store output
    out_ptr_offset = n_idx * stride_xn + c_idx * stride_xc + h_idx * stride_xh + w_idx * stride_xw
    tl.store(out_ptr + out_ptr_offset, out, mask=mask)


def triton_instance_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Triton-based Instance Normalization.
    """
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    n, c, h, w = x.shape
    
    # Allocate mean and variance tensors
    mean = torch.empty(n * c, dtype=torch.float32, device=x.device)
    var = torch.empty(n * c, dtype=torch.float32, device=x.device)
    
    # Launch reduction kernel
    reduce_grid = (n * c,)
    BLOCK_SIZE_REDUCE = 256
    instance_norm_reduce_kernel[reduce_grid](
        x, mean, var, n, c, h, w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_SIZE=BLOCK_SIZE_REDUCE,
    )
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Launch apply kernel
    total_elements = n * c * h * w
    BLOCK_SIZE_APPLY = 128
    apply_grid = (total_elements + BLOCK_SIZE_APPLY - 1) // BLOCK_SIZE_APPLY
    instance_norm_apply_kernel[apply_grid](
        x, mean, var, weight, bias, out, n, c, h, w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE_APPLY,
    )
    
    return out