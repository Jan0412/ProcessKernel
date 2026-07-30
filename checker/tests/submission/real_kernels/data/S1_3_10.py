import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    X, W, Bias, Out,
    stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_w_o, stride_w_i, stride_w_d, stride_w_h, stride_w_w,
    stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    N, IC, OC, ID, IH, IW, OD, OH, OW,
    kernel_size, stride, padding, output_padding,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Block indices
    pid_n = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Flatten the spatial and channel dimensions for the second program dimension
    # Each block handles a subset of N, OC, OD, OH, OW
    # We map pid_dhw to OC and OD, OH, OW
    
    # Calculate offsets for the block
    # We will iterate over IC and Kernel dims inside the kernel
    
    # Base pointers for this block
    # N offset
    n_off = pid_n * BLOCK_N
    # OC, OD, OH, OW offsets
    # We can flatten OC, OD, OH, OW into one dimension for pid_dhw if needed,
    # but let's just map pid_dhw to OD, OH, OW and loop OC?
    # Or map pid_dhw to OC and loop OD, OH, OW?
    # Let's map pid_dhw to a flattened index of (OD, OH, OW)
    # And loop OC? No, OC is large.
    
    # Let's use:
    # pid_n -> N
    # pid_dhw -> OC, OD, OH, OW
    
    total_spatial = OD * OH * OW
    # We want to cover OC * total_spatial
    # Let's say BLOCK_DHW = BLOCK_D * BLOCK_H * BLOCK_W
    # Let's say BLOCK_OC = 8
    
    # Let's define the offsets for the output block
    # This is a bit tricky. Let's simplify:
    # We will launch a grid that covers N x OC x OD x OH x OW
    # But 5D grid is not supported directly. We flatten.
    
    # Let's assume the grid is:
    # grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # grid_spatial = (OC * OD * OH * OW + BLOCK_OC * BLOCK_D * BLOCK_H * BLOCK_W - 1) // (BLOCK_OC * BLOCK_D * BLOCK_H * BLOCK_W)
    
    # Calculate OC, OD, OH, OW from pid_dhw
    block_id = pid_dhw
    BLOCK_DHW = BLOCK_D * BLOCK_H * BLOCK_W
    
    # This approach is getting complex. Let's do a simpler 2D grid:
    # Grid 0: N * OC
    # Grid 1: OD * OH * OW
    
    # Let's restart the grid mapping logic in the wrapper.
    # Here we assume:
    # pid_n -> Index in N * OC
    # pid_spatial -> Index in OD * OH * OW
    
    # Actually, let's just use a 2D grid in the wrapper:
    # grid[0] = N * OC
    # grid[1] = OD * OH * OW
    
    # But wait, the kernel signature has BLOCK_N and BLOCK_OC.
    # Let's use:
    # pid_n -> N
    # pid_oc -> OC
    # pid_spatial -> OD * OH * OW
    
    # This requires a 3D grid or flattening.
    # Let's flatten N and OC into pid_n_oc.
    # And OD, OH, OW into pid_spatial.
    
    # Let's implement a generic loop over IC and Kernel.
    
    # Offsets for N and OC
    # We need to extract N and OC from pid_n_oc
    pid_n_oc = pid_n
    n_idx = pid_n_oc // OC
    oc_idx = pid_n_oc % OC
    
    # Offsets for spatial
    pid_spatial = pid_dhw
    # We need to extract OD, OH, OW from pid_spatial
    # But we don't have OH, OW directly.
    # Let's pass OH, OW to the kernel? Yes.
    # Actually, let's just use od, oh, ow offsets.
    
    # Let's assume BLOCK_D, BLOCK_H, BLOCK_W are fixed.
    # We can calculate od, oh, ow from pid_spatial if we flatten OD*OH*OW.
    
    # Let's change the kernel to take:
    # pid_n_oc
    # pid_spatial
    
    # And compute:
    # n = pid_n_oc // OC
    # oc = pid_n_oc % OC
    
    # spatial_idx = pid_spatial
    # ow = spatial_idx % OW
    # oh = (spatial_idx // OW) % OH
    # od = spatial_idx // (OW * OH)
    
    # This is valid.
    
    n = pid_n_oc // OC
    oc = pid_n_oc % OC
    
    spatial_idx = pid_spatial
    ow = spatial_idx % OW
    oh = (spatial_idx // OW) % OH
    od = spatial_idx // (OW * OH)
    
    # Check bounds
    if n >= N or oc >= OC or od >= OD or oh >= OH or ow >= OW:
        return

    # Accumulator
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over IC and Kernel
    for ic in range(IC):
        # Base pointer for X[n, ic, :, :, :]
        x_ptr = X + n * stride_x_n + ic * stride_x_c
        
        # Iterate over Kernel
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Weight W[oc, ic, kd, kh, kw]
                    w_val = tl.load(W + oc * stride_w_o + ic * stride_w_i + kd * stride_w_d + kh * stride_w_h + kw * stride_w_w)
                    
                    # Input coordinates
                    # od = id * stride + kd - output_padding
                    # id = (od + output_padding - kd) / stride
                    
                    id_val = (od + output_padding - kd) // stride
                    ih_val = (oh + output_padding - kh) // stride
                    iw_val = (ow + output_padding - kw) // stride
                    
                    # Check if exact division
                    if (od + output_padding - kd) % stride == 0 and \
                       (oh + output_padding - kh) % stride == 0 and \
                       (ow + output_padding - kw) % stride == 0:
                        
                        # Check bounds for ID, IH, IW
                        if 0 <= id_val < ID and 0 <= ih_val < IH and 0 <= iw_val < IW:
                            x_val = tl.load(x_ptr + id_val * stride_x_d + ih_val * stride_x_h + iw_val * stride_x_w)
                            acc += x_val * w_val
    
    # Add bias if present
    if Bias is not None:
        bias_val = tl.load(Bias + oc)
        acc += bias_val
    
    # Store output
    out_ptr = Out + n * stride_out_n + oc * stride_out_c + od * stride_out_d + oh * stride_out_h + ow * stride_out_w
    tl.store(out_ptr, acc)


def triton_conv_transpose3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                            stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, kernel_size: int = 1):
    """
    Wrapper for the Triton ConvTranspose3d kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert groups == 1, "Only groups=1 is supported."
    
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    assert KD == KH == KW == kernel_size, "Kernel must be square and match kernel_size."
    
    # Calculate output shape
    OD = (ID - 1) * stride - 2 * padding + KD + output_padding
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding
    
    out = torch.empty((N, OC, OD, OH, OW), dtype=x.dtype, device=x.device)
    
    if bias is not None:
        assert bias.shape[0] == OC, "Bias shape mismatch."
    
    # Grid configuration
    # We flatten N and OC into one dimension, and OD, OH, OW into another.
    # This allows us to use a 2D grid.
    
    BLOCK_N = 1
    BLOCK_OC = 1
    BLOCK_D = 1
    BLOCK_H = 1
    BLOCK_W = 1
    
    # Total blocks for N*OC
    grid_n_oc = N * OC
    # Total blocks for OD*OH*OW
    grid_spatial = OD * OH * OW
    
    grid = (grid_n_oc, grid_spatial)
    
    # Strides
    stride_x_n = x.stride(0)
    stride_x_c = x.stride(1)
    stride_x_d = x.stride(2)
    stride_x_h = x.stride(3)
    stride_x_w = x.stride(4)
    
    stride_w_o = weight.stride(0)
    stride_w_i = weight.stride(1)
    stride_w_d = weight.stride(2)
    stride_w_h = weight.stride(3)
    stride_w_w = weight.stride(4)
    
    stride_out_n = out.stride(0)
    stride_out_c = out.stride(1)
    stride_out_d = out.stride(2)
    stride_out_h = out.stride(3)
    stride_out_w = out.stride(4)
    
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
        stride_w_o, stride_w_i, stride_w_d, stride_w_h, stride_w_w,
        stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
        N, IC, OC, ID, IH, IW, OD, OH, OW,
        kernel_size, stride, padding, output_padding,
        BLOCK_N=BLOCK_N, BLOCK_OC=BLOCK_OC, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels // groups, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias_param = None
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias_param,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding,
            groups=self.groups, kernel_size=self.kernel_size
        )