import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    stride_xb,
    stride_xf,
    stride_xd1,
    stride_xd2,
    num_features,
    dim2,
    eps,
    BLOCK_D2: tl.constexpr,
):
    # Get program IDs
    b_idx = tl.program_id(0)
    d1_idx = tl.program_id(1)
    d2_block_id = tl.program_id(2)
    
    # Compute offsets for D2 dimension
    d2_offsets = d2_block_id * BLOCK_D2 + tl.arange(0, BLOCK_D2)
    mask = d2_offsets < dim2
    
    # Base offset for the current (b_idx, d1_idx) slice
    # We are accessing x[b_idx, :, d1_idx, d2_offsets]
    # Strides: B is largest, then F, then D1, then D2
    
    # Pointer to the start of the block for a specific feature f
    # We need to loop over f
    
    # Accumulator for sum of squares
    # Shape: (BLOCK_D2,)
    sum_sq = tl.zeros([BLOCK_D2], dtype=tl.float32)
    
    # Loop over features to compute RMS
    for f in range(num_features):
        # Calculate pointer offset for current feature
        # offset = b_idx * stride_xb + f * stride_xf + d1_idx * stride_xd1
        # But we can factor out constant parts
        # Actually, simpler to compute full offset
        
        # x_ptr + b_idx * stride_xb + d1_idx * stride_xd1 + f * stride_xf + d2_offsets
        # Let's compute the base pointer for (b_idx, d1_idx)
        base_offset = b_idx * stride_xb + d1_idx * stride_xd1
        
        # Pointer for feature f
        f_offset = base_offset + f * stride_xf
        
        # Load data
        x_block = tl.load(x_ptr + f_offset + d2_offsets, mask=mask, other=0.0)
        
        # Accumulate square
        sum_sq += x_block * x_block
        
    # Compute RMS
    # rms = sqrt(mean(x^2) + eps) = sqrt(sum_sq / num_features + eps)
    rms = tl.sqrt(sum_sq / num_features + eps)
    
    # Loop over features again to normalize and store
    for f in range(num_features):
        base_offset = b_idx * stride_xb + d1_idx * stride_xd1
        f_offset = base_offset + f * stride_xf
        
        # Load original data again
        x_block = tl.load(x_ptr + f_offset + d2_offsets, mask=mask, other=0.0)
        
        # Normalize
        out_block = x_block / rms
        
        # Store
        tl.store(out_ptr + f_offset + d2_offsets, out_block, mask=mask)

def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5):
    # x shape: (B, F, D1, D2)
    B, F, D1, D2 = x.shape
    
    # Output tensor
    out = torch.empty_like(x)
    
    # Strides
    stride_xb = x.stride(0)
    stride_xf = x.stride(1)
    stride_xd1 = x.stride(2)
    stride_xd2 = x.stride(3) # Should be 1
    
    # Block size for D2
    BLOCK_D2 = 128
    
    # Grid configuration
    # We map (B, D1, D2_blocks) to programs
    grid = (B, D1, triton.cdiv(D2, BLOCK_D2))
    
    # Launch kernel
    rms_norm_kernel[grid](
        x, out,
        stride_xb, stride_xf, stride_xd1, stride_xd2,
        F, D2, eps,
        BLOCK_D2=BLOCK_D2
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return triton_rms_norm(x, self.eps)