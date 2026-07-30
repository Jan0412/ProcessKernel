import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose_3d_kernel(
    X, W, B, Out,
    N, C_in, C_out, D, H, W_dim,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    ODP, ODH, ODW,
    Groups,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate global index
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total number of output elements
    D_out = (D - 1) * SD + KD - 2 * PD + ODP
    H_out = (H - 1) * SH + KH - 2 * PH + ODH
    W_out = (W_dim - 1) * SW + KW - 2 * PW + ODW
    total_out = N * C_out * D_out * H_out * W_out
    
    # Mask for valid output elements
    mask = offset < total_out
    
    # Decode output coordinates from flat index
    # idx = n * (C_out * D_out * H_out * W_out) + c_out * (D_out * H_out * W_out) + ...
    rem = offset
    
    # W_out
    wo = rem % W_out
    rem = rem // W_out
    # H_out
    ho = rem % H_out
    rem = rem // H_out
    # D_out
    do = rem % D_out
    rem = rem // D_out
    # C_out
    co = rem % C_out
    # N
    n = rem // C_out
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                # Calculate input coordinates
                # do = di * SD + kd - PD + ODP  => di = (do - kd + PD - ODP) / SD
                # ho = hi * SH + kh - PH + ODH  => hi = (ho - kh + PH - ODH) / SH
                # wo = wi * SW + kw - PW + ODW  => wi = (wo - kw + PW - ODW) / SW
                
                di_num = do - kd + PD - ODP
                hi_num = ho - kh + PH - ODH
                wi_num = wo - kw + PW - ODW
                
                # Check if division is exact
                valid_d = (di_num >= 0) & (di_num % SD == 0)
                valid_h = (hi_num >= 0) & (hi_num % SH == 0)
                valid_w = (wi_num >= 0) & (wi_num % SW == 0)
                
                di = di_num // SD
                hi = hi_num // SH
                wi = wi_num // SW
                
                # Check input bounds
                valid_bounds = (di < D) & (hi < H) & (wi < W_dim)
                valid = valid_d & valid_h & valid_w & valid_bounds
                
                # Determine input channel group
                # c_out maps to c_in via groups
                # c_in = (c_out % Groups) * (C_in // Groups) + ... 
                # Actually, for groups, c_out // Groups is the group index.
                # c_in iterates over the channels in that group.
                # We need to loop over input channels in the group.
                
                # Triton loop over channels is hard to unroll if C_in is large.
                # We will loop over input channels inside the kernel.
                # However, doing a python loop over C_in in Triton is slow if C_in is large.
                # Better approach: The kernel above is a "naive" implementation.
                # For a robust Triton kernel, we usually tile.
                # Given the constraints of a single snippet, we will iterate channels using a loop.
                # To optimize, we can assume the user might have small kernels.
                
                # Let's refine the channel loop.
                # We need to access W[c_out, c_in // Groups, kd, kh, kw]
                # And X[n, c_in, di, hi, wi]
                
                # Since we are inside the spatial loop, we need to load X and W for all valid channels.
                # This is difficult to vectorize efficiently in a simple kernel without complex tiling.
                # We will stick to a simpler accumulation that might be slower than cuDNN but demonstrates the custom kernel.
                # A better way for Triton is to tile the C_in dimension.
                
                # Let's restart the kernel strategy for better performance:
                # Instead of iterating output, let's iterate input? No, output is standard.
                # Let's use a loop over input channels.
                pass

    # Since writing a fully tiled 3D ConvTranspose in a single block is complex,
    # we will use a simpler approach: Iterate over output, and for each output,
    # loop over the kernel window and input channels.
    # This is O(K^3 * C_in) per output element.
    
    # Reset acc
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over input channels
    # We can only do this if we can load X and W efficiently.
    # We will load W for the current [co, c_in, kd, kh, kw]
    # and X for [n, c_in, di, hi, wi]
    
    # Calculate group info
    c_in_per_group = C_in // Groups
    group_id = co // c_in_per_group
    
    for c_in in range(C_in):
        # Check if c_in belongs to the same group as c_out
        c_in_group = c_in // c_in_per_group
        if c_in_group != group_id:
            continue
            
        # Iterate kernel window
        for kd in range(KD):
            for kh in range(KH):
                for kw in range(KW):
                    di_num = do - kd + PD - ODP
                    hi_num = ho - kh + PH - ODH
                    wi_num = wo - kw + PW - ODW
                    
                    valid_d = (di_num >= 0) & (di_num % SD == 0)
                    valid_h = (hi_num >= 0) & (hi_num % SH == 0)
                    valid_w = (wi_num >= 0) & (wi_num % SW == 0)
                    
                    di = di_num // SD
                    hi = hi_num // SH
                    wi = wi_num // SW
                    
                    valid_bounds = (di < D) & (hi < H) & (wi < W_dim)
                    valid = valid_d & valid_h & valid_w & valid_bounds
                    
                    # Load Weight
                    # W shape: [C_out, C_in // Groups, KD, KH, KW]
                    w_idx = co * (C_in // Groups) * KD * KH * KW + \
                            (c_in % c_in_per_group) * KD * KH * KW + \
                            kd * KH * KW + kh * KW + kw
                    w_val = tl.load(W + w_idx)
                    
                    # Load Input
                    # X shape: [N, C_in, D, H, W]
                    x_idx = n * C_in * D * H * W_dim + \
                            c_in * D * H * W_dim + \
                            di * H * W_dim + hi * W_dim + wi
                    x_val = tl.load(X + x_idx, mask=valid & mask, other=0.0)
                    
                    acc += x_val * w_val

    # Add bias
    if B is not None:
        bias_val = tl.load(B + co, mask=mask, other=0.0)
        acc += bias_val

    # Store output
    tl.store(Out + offset, acc, mask=mask)

def triton_conv_transpose_3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1):
    x = x.contiguous()
    weight = weight.contiguous()
    
    N, C_in, D, H, W = x.shape
    C_out, C_in_per_group, KD, KH, KW = weight.shape
    
    SD, SH, SW = stride
    PD, PH, PW = padding
    ODP, ODH, ODW = output_padding
    
    D_out = (D - 1) * SD + KD - 2 * PD + ODP
    H_out = (H - 1) * SH + KH - 2 * PH + ODH
    W_out = (W - 1) * SW + KW - 2 * PW + ODW
    
    out = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 128
    total_out = N * C_out * D_out * H_out * W_out
    grid = (triton.cdiv(total_out, BLOCK_SIZE),)
    
    conv_transpose_3d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out, D, H, W,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        ODP, ODH, ODW,
        groups,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters to initialize the weight and bias tensors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        
        # Initialize weight and bias like nn.ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose_3d(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups
        )