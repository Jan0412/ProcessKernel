import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def post_conv_logsumexp_kernel(
    x_conv_ptr,      # Pointer to the output of Conv2d [B, C, H, W]
    gn_weight_ptr,   # Pointer to GroupNorm weight [C]
    gn_bias_ptr,     # Pointer to GroupNorm bias [C]
    out_ptr,         # Pointer to output [B, 1, H, W]
    
    C,               # Number of channels
    H,               # Height
    W,               # Width
    G,               # Number of groups
    eps,             # Epsilon for GroupNorm
    
    stride_xc, stride_xh, stride_xw,
    stride_oc, stride_oh, stride_ow,
    
    BLOCK_SIZE: tl.constexpr
):
    # Each program instance handles one spatial location (h, w)
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    if pid_h >= H or pid_w >= W:
        return

    # Initialize accumulators for GroupNorm stats
    # We need to iterate over all channels to compute stats
    # Since channels are contiguous in memory, we can vectorize
    
    # Offsets for channel iteration
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Variables to store group statistics
    # We will compute stats for each group and store them in registers
    # Max groups is small (16), so this fits easily in registers
    group_means = tl.zeros([G], dtype=tl.float32)
    group_vars = tl.zeros([G], dtype=tl.float32)
    
    # Pass 1: Compute Mean and Variance
    # Iterate over all channels
    for c_start in range(0, C, BLOCK_SIZE):
        c_offsets = c_start + offsets
        mask = c_offsets < C
        
        # Load x_conv values for this (h, w) across channels
        # x_conv layout is [C, H, W], so we stride over C
        x_ptrs = x_conv_ptr + c_offsets * stride_xc + pid_h * stride_xh + pid_w * stride_xw
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Determine group index for each channel
        # group_idx = c // (C // G)
        # We can compute this using integer division
        # However, to keep it efficient, we can just loop G times? 
        # No, iterating channels is better.
        
        # Calculate group index for the current block of channels
        # We can use a helper or just compute per-element
        # group_id = tl.floor_div(c_offsets, C // G)
        
        # To simplify and ensure correctness with Triton's reduction semantics,
        # let's compute sum and sum_sq for all elements, then aggregate by group.
        # Actually, Triton's tl.sum reduces the vector. 
        # We need sum per group.
        
        # Let's do a simpler approach: Loop over groups? 
        # If C is large, looping groups is better.
        # But we are inside a loop over channels.
        
        # Let's stick to channel loop and accumulate into a temporary buffer?
        # Registers are limited. Let's just accumulate directly if G is small.
        
        # Alternative: Since we are in a single thread, we can just loop channels and add to the specific group index.
        # But we are processing a vector of channels.
        
        # Let's use a different strategy for Pass 1:
        # Iterate over groups, and inside, iterate over channels in that group.
        pass

    # Restarting logic for cleaner register usage:
    # Iterate over Groups
    group_size = C // G
    
    # We will launch a loop over groups
    # Since G is small (16), this is very fast.
    
    # We need to store the normalized values or recompute them in Pass 2.
    # Recomputing is memory efficient but compute heavy. Given FP32 and simple ops, 
    # recomputing is fine.
    
    # Pass 1: Compute stats per group
    for g in range(G):
        # Channels for this group
        c_start_g = g * group_size
        
        sum_val = 0.0
        sum_sq_val = 0.0
        
        # Iterate over channels in this group
        for c_off in range(0, group_size, BLOCK_SIZE):
            c_abs = c_start_g + c_off + offsets
            mask = c_abs < C
            
            # Load
            x_ptrs = x_conv_ptr + c_abs * stride_xc + pid_h * stride_xh + pid_w * stride_xw
            x = tl.load(x_ptrs, mask=mask, other=0.0)
            
            sum_val += tl.sum(x)
            sum_sq_val += tl.sum(x * x)
            
        # Compute mean and var
        # tl.sum reduces the vector to a scalar
        # Note: sum_val is actually the sum of the vector sums.
        
        mean = sum_val / group_size
        var = sum_sq_val / group_size - mean * mean
        # Add epsilon for stability
        var = var + eps
        
        # Store in "registers" (we can just use a list or recompute if needed)
        # Since G is small, we can store these in a small array
        group_means[g] = mean
        group_vars[g] = var

    # Pass 2: Normalize, Activate, Residual, Accumulate for LogSumExp
    logsumexp_accum = 0.0
    
    for g in range(G):
        mean = group_means[g]
        var = group_vars[g]
        std = tl.sqrt(var)
        inv_std = 1.0 / std
        
        c_start_g = g * group_size
        
        for c_off in range(0, group_size, BLOCK_SIZE):
            c_abs = c_start_g + c_off + offsets
            mask = c_abs < C
            
            # Load x_conv
            x_ptrs = x_conv_ptr + c_abs * stride_xc + pid_h * stride_xh + pid_w * stride_xw
            x = tl.load(x_ptrs, mask=mask, other=0.0)
            
            # Load GN weight and bias
            w = tl.load(gn_weight_ptr + c_abs, mask=mask, other=1.0)
            b = tl.load(gn_bias_ptr + c_abs, mask=mask, other=0.0)
            
            # Group Norm
            x_norm = (x - mean) * inv_std
            x_affine = x_norm * w + b
            
            # Tanh
            x_tanh = tl.math.tanh(x_affine)
            
            # HardSwish: x * max(0, min(3, x+3)) / 6
            # Note: PyTorch HardSwish is x * ReLU6(x + 3) / 6
            x_hs = x_tanh * tl.minimum(3.0, tl.maximum(0.0, x_tanh + 3.0)) / 6.0
            
            # Residual Add: x_conv + x_hard_swish
            x_res = x + x_hs
            
            # LogSumExp Accumulation: sum(exp(x_res))
            # We accumulate exp(x_res) in a running sum
            logsumexp_accum += tl.sum(tl.exp(x_res))
            
    # Final Log
    final_val = tl.log(logsumexp_accum)
    
    # Store output
    # Output layout [B, 1, H, W]
    out_ptrs = out_ptr + pid_h * stride_oh + pid_w * stride_ow
    tl.store(out_ptrs, final_val)


def post_conv_module(x_conv: torch.Tensor, gn: nn.GroupNorm) -> torch.Tensor:
    """
    Fused Triton kernel for GroupNorm, Tanh, HardSwish, Add, and LogSumExp.
    
    Args:
        x_conv: Output of the Conv2d layer [B, C, H, W]
        gn: The GroupNorm layer
        
    Returns:
        Output tensor [B, 1, H, W]
    """
    assert x_conv.is_cuda
    B, C, H, W = x_conv.shape
    G = gn.num_groups
    eps = gn.eps
    
    # Prepare output tensor
    out = torch.empty((B, 1, H, W), dtype=x_conv.dtype, device=x_conv.device)
    
    # Ensure inputs are contiguous
    x_conv = x_conv.contiguous()
    gn_weight = gn.weight.contiguous()
    gn_bias = gn.bias.contiguous()
    
    # Block size for channel vectorization
    BLOCK_SIZE = 64
    
    # Grid dimensions: H x W
    grid = (H, W)
    
    # Strides
    stride_xc = x_conv.stride(1)
    stride_xh = x_conv.stride(2)
    stride_xw = x_conv.stride(3)
    
    stride_oc = out.stride(1)
    stride_oh = out.stride(2)
    stride_ow = out.stride(3)
    
    # Launch kernel for each batch item
    for i in range(B):
        # Slice pointers for the current batch
        x_conv_batch_ptr = x_conv_ptr = x_conv[i].data_ptr()
        out_batch_ptr = out[i].data_ptr()
        
        post_conv_logsumexp_kernel[grid](
            x_conv_batch_ptr,
            gn_weight_ptr=gn_weight.data_ptr(),
            gn_bias_ptr=gn_bias.data_ptr(),
            out_ptr=out_batch_ptr,
            
            C=C, H=H, W=W, G=G, eps=eps,
            
            stride_xc=stride_xc, stride_xh=stride_xh, stride_xw=stride_xw,
            stride_oc=stride_oc, stride_oh=stride_oh, stride_ow=stride_ow,
            
            BLOCK_SIZE=BLOCK_SIZE
        )
        
    return out


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp.
    Optimized with a fused Triton kernel for the post-conv operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        # Tanh and HardSwish are fused into the Triton kernel, kept for reference/compatibility if needed
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        # Convolution (kept as PyTorch/nn.Conv2d for performance)
        x_conv = self.conv(x)
        
        # Fused Post-Conv: GroupNorm + Tanh + HardSwish + Add + LogSumExp
        x_logsumexp = post_conv_module(x_conv, self.group_norm)
        
        return x_logsumexp