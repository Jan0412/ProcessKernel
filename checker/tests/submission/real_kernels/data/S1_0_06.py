import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_transpose_kernel(
    X_ptr, W_ptr, O_ptr, B_ptr,
    G, IC, IC_out, OC, OC, OC,
    BLOCK_M, BLOCK_N, BLOCK_K, K, PAD, OFF_PAD, GROUPS, TORCH_DEVICES
):
    # Grid coordinates
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Base indices for the tile
    base_n = pid_n * BLOCK_M
    base_m = pid_m * BLOCK_N

    # Offsets for the output tile
    off_n = base_n + tl.arange(0, BLOCK_M)
    off_m = base_m + tl.arange(0, BLOCK_N)

    # Mask for output dimensions
    mask_n = off_n < IC_out
    mask_m = off_m < OC

    # Accumulator for the tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over input channels per group and kernel size
    # IC_in_per_group = IC // GROUPS
    IC_in_per_group = IC // GROUPS
    
    # We iterate over blocks of input channels
    for block_k in range(0, (IC_in_per_group + BLOCK_K - 1) // BLOCK_K):
        # Loop over kernel dimension
        for k in range(0, K):
            # Calculate input channel indices
            # c_in = block_k * BLOCK_K + k_idx
            # But we must respect groups: c_in = (off_n % GROUPS) + block_k * BLOCK_K + k_idx
            # Wait, the standard layout is:
            # W[IC_out, IC_in // GROUPS, K]
            # So for a given N (IC_out), the corresponding IC_in is:
            # c_in = block_k * BLOCK_K + k_idx
            
            # Let's compute the actual c_in indices for the tile
            # off_n contains IC_out indices. 
            # c_in = (off_n % GROUPS) + block_k * BLOCK_K + tl.arange(0, BLOCK_K)
            # This is not contiguous in memory if GROUPS > 1 and BLOCK_M spans multiple groups.
            # However, PyTorch's ConvTranspose1d stores W as (IC_out, IC_in // GROUPS, K).
            # So W[IC_out, c_in, k] is contiguous in IC_out and c_in!
            # We can just load W[off_n, block_k * BLOCK_K + k_idx, k]
            
            off_k = tl.arange(0, BLOCK_K)
            c_in = block_k * BLOCK_K + off_k
            
            # Mask for c_in
            mask_c_in = c_in < IC_in_per_group
            
            # Load W tile: shape (BLOCK_M, BLOCK_K)
            # W is (IC_out, IC_in // GROUPS, K)
            # We need to flatten or index correctly.
            # W_ptr + off_n[:, None] * (IC_in // GROUPS) * K + c_in[None, :] * K + k
            # But W is contiguous in (IC_out, IC_in // GROUPS, K).
            # So stride for IC_out is (IC_in // GROUPS) * K
            # Stride for IC_in // GROUPS is K
            # Stride for K is 1
            
            w_offsets = off_n[:, None] * (IC_in_per_group * K) + c_in[None, :] * K + k
            w = tl.load(W_ptr + w_offsets, mask=mask_n[:, None] & mask_c_in[None, :], other=0.0)
            
            # Load X tile: shape (BLOCK_M, BLOCK_K)
            # X is (G, IC, OC)
            # x_idx = (off_m + 2 * PAD - k - 1 - OFF_PAD) // STRIDE
            # We need to check if (off_m + 2 * PAD - k - 1 - OFF_PAD) % STRIDE == 0
            
            numerator = off_m + 2 * PAD - k - 1 - OFF_PAD
            rem = numerator % STRIDE
            x_idx = numerator // STRIDE
            
            # Mask for x_idx
            mask_x = mask_m & (rem == 0) & (x_idx >= 0) & (x_idx < OC)
            
            # X_offsets: G * IC * OC + c_in * OC + x_idx
            # But c_in is (BLOCK_M, BLOCK_K) and x_idx is (BLOCK_N,)
            # We need (BLOCK_M, BLOCK_N)
            x_offsets = pid_b * IC * OC + c_in[:, None] * OC + x_idx[None, :]
            x = tl.load(X_ptr + x_offsets, mask=mask_n[:, None] & mask_x[None, :], other=0.0)
            
            # Accumulate
            acc = acc + tl.where(mask_n[:, None] & mask_c_in[None, :] & mask_x[None, :], w * x, 0.0)

    # Add bias
    if B_ptr is not None:
        bias = tl.load(B_ptr + off_n, mask=mask_n, other=0.0)
        acc = acc + bias[:, None]
    
    # Store output
    o_offsets = pid_b * IC_out * OC + off_n[:, None] * OC + off_m[None, :]
    tl.store(O_ptr + o_offsets, acc, mask=mask_n[:, None] & mask_m[None, :])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.kernel_size = kernel_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        out_c = self.conv1d_transpose.weight.shape[0]
        out_l = (l - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        # Prepare output tensor
        out = torch.empty((b, out_c, out_l), device=x.device, dtype=x.dtype)
        
        # Get weights and bias
        w = self.conv1d_transpose.weight
        b_ptr = self.conv1d_transpose.bias if self.conv1d_transpose.bias is not None else None
        
        # Kernel parameters
        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 8
        
        # Grid
        grid = (
            (out_c + BLOCK_M - 1) // BLOCK_M,
            (out_l + BLOCK_N - 1) // BLOCK_N,
            b
        )
        
        conv1d_transpose_kernel[grid](
            x, w, out, b_ptr,
            b, c, out_c, out_l, out_l, out_l,
            BLOCK_M, BLOCK_N, BLOCK_K, self.kernel_size, self.padding, self.output_padding, self.groups, x.device
        )
        
        return out