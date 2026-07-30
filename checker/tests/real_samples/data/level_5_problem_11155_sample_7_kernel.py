import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def gn_relu_kernel(
    x_ptr,  # Input tensor
    mean_ptr,  # Mean tensor (per group)
    var_ptr,  # Variance tensor (per group)
    weight_ptr,  # GroupNorm weight
    bias_ptr,  # GroupNorm bias
    out_ptr,  # Output tensor
    n_elements,  # Total number of elements
    D: tl.constexpr,  # Channels per group
    C: tl.constexpr,  # Total channels
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel assumes input is already normalized with per-group statistics
    # and applies GroupNorm + ReLU fusion
    
    # Calculate which channel this thread is processing
    channel_id = tl.program_id(0)
    
    # Offset to start of this channel's data
    channel_stride = n_elements // C
    sample_start = channel_id * channel_stride
    
    # Load mean and variance for this group
    group_id = channel_id // D
    mean_val = tl.load(mean_ptr + group_id)
    var_val = tl.load(var_ptr + group_id)
    
    # Load weight and bias for this channel
    w = tl.load(weight_ptr + channel_id)
    b = tl.load(bias_ptr + channel_id)
    
    # Process elements in this channel
    for i in range(BLOCK_SIZE):
        offset = sample_start + i
        if offset < n_elements:
            # Load input
            x = tl.load(x_ptr + offset)
            # Normalize: (x - mean) / sqrt(var + eps)
            normalized = (x - mean_val) * rsqrt(var_val + eps)
            # Apply scale and shift
            out = normalized * w + b
            # Apply ReLU
            out = tl.where(out > 0, out, 0.0)
            # Store result
            tl.store(out_ptr + offset, out)


@triton.jit
def fused_add_relu_kernel(
    x_ptr,  # First input (residual)
    y_ptr,  # Second input (transformed)
    out_ptr,  # Output
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    out = x + y
    out = tl.where(out > 0, out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def group_norm_relu_forward(x, weight, bias, mean, var, eps=1e-5):
    """Apply GroupNorm + ReLU using Triton kernel"""
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    C = x.size(1)  # Number of channels
    # Channels per group (assuming groups = C/D)
    D = weight.numel()  # This should equal C since weight is per-channel
    BLOCK_SIZE = 256
    
    grid = (C,)  # One block per channel
    
    gn_relu_kernel[grid](
        x, mean, var, weight, bias, out,
        n_elements, D, C, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


def fused_add_relu(x, y):
    """Add two tensors and apply ReLU"""
    x = x.contiguous()
    y = y.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    fused_add_relu_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, dim):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, dim, eps=0.0001)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, dim, eps=0.0001)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, dim, eps=0.0001)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, dim, eps=0.0001)

    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        residual = x
        
        # First conv block
        out = self.conv1(x)
        
        # Compute GroupNorm statistics for bn1
        # GroupNorm with 2 groups: split channels into 2 groups
        B, C, H, W = out.shape
        G = 2
        D = C // G  # Channels per group
        if D == 0:
            D = 1  # Handle edge case
            
        # Reshape for group normalization: (B, G, D, H, W)
        x_gn1 = out.view(B, G, D, H, W)
        mean1 = x_gn1.mean(dim=[0, 3, 4], keepdim=True)  # Per-group mean
        var1 = x_gn1.var(dim=[0, 3, 4], keepdim=True, unbiased=False)  # Per-group variance
        
        # Flatten mean and var for kernel
        mean1_flat = mean1.view(C)
        var1_flat = var1.view(C)
        
        out = group_norm_relu_forward(out, self.bn1.weight, self.bn1.bias, 
                                     mean1_flat, var1_flat, self.bn1.eps)
        
        # Second conv block
        out = self.conv2(out)
        
        # Compute GroupNorm statistics for bn2
        x_gn2 = out.view(B, G, D, H, W)
        mean2 = x_gn2.mean(dim=[0, 3, 4], keepdim=True)
        var2 = x_gn2.var(dim=[0, 3, 4], keepdim=True, unbiased=False)
        
        mean2_flat = mean2.view(C)
        var2_flat = var2.view(C)
        
        # Apply GroupNorm without ReLU first
        out = group_norm_forward(out, self.bn2.weight, self.bn2.bias, 
                                mean2_flat, var2_flat, self.bn2.eps)
        
        # Fused add + ReLU
        out = fused_add_relu(out, residual)
        
        return out


def group_norm_forward(x, weight, bias, mean, var, eps=1e-5):
    """Apply GroupNorm using Triton kernel (without ReLU)"""
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    C = x.size(1)
    D = C // 2 if weight.numel() == C else weight.numel()
    BLOCK_SIZE = 256
    
    grid = (C,)
    
    # Create a kernel without ReLU for this case
    @triton.jit
    def gn_only_kernel(
        x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
        n_elements, D: tl.constexpr, C: tl.constexpr, eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        channel_id = tl.program_id(0)
        channel_stride = n_elements // C
        sample_start = channel_id * channel_stride
        group_id = channel_id // D
        mean_val = tl.load(mean_ptr + group_id)
        var_val = tl.load(var_ptr + group_id)
        w = tl.load(weight_ptr + channel_id)
        b = tl.load(bias_ptr + channel_id)
        
        for i in range(BLOCK_SIZE):
            offset = sample_start + i
            if offset < n_elements:
                x_val = tl.load(x_ptr + offset)
                normalized = (x_val - mean_val) * rsqrt(var_val + eps)
                out = normalized * w + b
                tl.store(out_ptr + offset, out)
    
    gn_only_kernel[grid](
        x, mean, var, weight, bias, out,
        n_elements, D, C, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


# Helper function for rsqrt
@triton.jit
def rsqrt(x):
    return 1.0 / tl.sqrt(x)