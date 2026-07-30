import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    X, W, Bias, Out,
    stride_x_n, stride_x_c, stride_x_w, stride_x_h, stride_x_d,
    stride_w_c, stride_w_kc, stride_w_kw, stride_w_kh, stride_w_kd,
    stride_o_n, stride_o_c, stride_o_w, stride_o_h, stride_o_d,
    N, C_in, C_out,
    W, H, D,
    W_out, H_out, D_out,
    KW, KH, KD,
    pad_w, pad_h, pad_d,
    stride_w, stride_h, stride_d,
    dilation_w, dilation_h, dilation_d,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Each program handles a contiguous block of output pixels
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode linear offset into N, C_out, W_out, H_out, D_out
    # We process multiple output pixels in parallel within one block
    # To simplify, we will compute coordinates for each offset in the block
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # We need to iterate over input channels and kernel volume for each output pixel.
    # To keep Triton efficient, we loop over channels and kernel dims outside the vectorized offset logic.
    # This means we will load slices of X and W.
    
    # Calculate base coordinates for the current block's offsets
    # Since offsets are 1D, we decode them.
    # off_n = offsets // (C_out * W_out * H_out * D_out)
    # off_c_out = (offsets // (W_out * H_out * D_out)) % C_out
    # off_w_out = (offsets // (H_out * D_out)) % W_out
    # off_h_out = (offsets // D_out) % H_out
    # off_d_out = offsets % D_out
    
    # Precompute divisors
    div_c_out = W_out * H_out * D_out
    div_w_out = H_out * D_out
    div_h_out = D_out
    
    off_n = offsets // (C_out * div_c_out)
    off_c_out = (offsets // div_c_out) % C_out
    off_w_out = (offsets // div_w_out) % W_out
    off_h_out = (offsets // div_h_out) % H_out
    off_d_out = offsets % D_out

    # Compute pointers to the specific output channel weights
    # W layout: (C_out, C_in/groups, KW, KH, KD) -> simplified to (C_out, C_in, KW, KH, KD) for groups=1
    # We only need weights for the current C_out
    w_base = W + off_c_out * stride_w_c
    
    # Iterate over kernel volume
    for kw in range(KW):
        for kh in range(KH):
            for kd in range(KD):
                # Compute input spatial coordinates
                # x_in = stride * out + dilation * k - pad
                in_w = off_w_out * stride_w + kw * dilation_w - pad_w
                in_h = off_h_out * stride_h + kh * dilation_h - pad_h
                in_d = off_d_out * stride_d + kd * dilation_d - pad_d

                # Check bounds for input spatial dimensions
                # If out of bounds, we load 0.0
                mask_spatial = (in_w >= 0) & (in_w < W) & \
                               (in_h >= 0) & (in_h < H) & \
                               (in_d >= 0) & (in_d < D)

                # Compute input pointer base for this spatial location
                # X layout: (N, C_in, W, H, D)
                x_ptr = X + off_n * stride_x_n + in_w * stride_x_w + in_h * stride_x_h + in_d * stride_x_d
                
                # Weight pointer for this kernel position and output channel
                w_ptr = w_base + kw * stride_w_kw + kh * stride_w_kh + kd * stride_w_kd
                
                # Iterate over input channels
                for c in range(C_in):
                    # Load input slice: [BLOCK_SIZE] elements across N, C, spatial
                    # Actually, x_ptr already has N, W, H, D fixed. We just add channel offset
                    x_val = tl.load(x_ptr + c * stride_x_c, mask=mask_spatial & mask, other=0.0)
                    
                    # Load weight: scalar for this C_out, C_in, KW, KH, KD
                    w_val = tl.load(w_ptr + c * stride_w_c, mask=mask, other=0.0)
                    
                    acc += x_val * w_val

    # Add bias if present
    if HAS_BIAS:
        bias_val = tl.load(Bias + off_c_out, mask=mask, other=0.0)
        acc += bias_val

    # Compute output pointer
    out_ptr = Out + off_n * stride_o_n + off_c_out * stride_o_c + \
              off_w_out * stride_o_w + off_h_out * stride_o_h + off_d_out * stride_o_d
    
    # Store result
    tl.store(out_ptr, acc, mask=mask)


def conv3d_triton(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Custom Triton 3D convolution.
    Assumes groups=1 for simplicity in this optimized path.
    """
    assert groups == 1, "Only groups=1 is supported in this optimized Triton kernel."
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, W, H, D = x.shape
    C_out, _, KW, KH, KD = weight.shape
    
    # Calculate output dimensions
    W_out = (W + 2 * padding[0] - dilation[0] * (KW - 1) - 1) // stride[0] + 1
    H_out = (H + 2 * padding[1] - dilation[1] * (KH - 1) - 1) // stride[1] + 1
    D_out = (D + 2 * padding[2] - dilation[2] * (KD - 1) - 1) // stride[2] + 1
    
    out = torch.empty((N, C_out, W_out, H_out, D_out), device=x.device, dtype=torch.float32)
    
    n_elements = N * C_out * W_out * H_out * D_out
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Strides for X (N, C, W, H, D)
    sx_n = C_in * W * H * D
    sx_c = W * H * D
    sx_w = H * D
    sx_h = D
    sx_d = 1

    # Strides for W (C_out, C_in, KW, KH, KD)
    sw_c = C_in * KW * KH * KD
    sw_kc = KW * KH * KD
    sw_kw = KH * KD
    sw_kh = KD
    sw_kd = 1

    # Strides for Out (N, C_out, W_out, H_out, D_out)
    so_n = C_out * W_out * H_out * D_out
    so_c = W_out * H_out * D_out
    so_w = H_out * D_out
    so_h = D_out
    so_d = 1

    conv3d_kernel[grid](
        x, weight, bias, out,
        sx_n, sx_c, sx_w, sx_h, sx_d,
        sw_c, sw_kc, sw_kw, sw_kh, sw_kd,
        so_n, so_c, so_w, so_h, so_d,
        N, C_in, C_out,
        W, H, D,
        W_out, H_out, D_out,
        KW, KH, KD,
        padding[0], padding[1], padding[2],
        stride[0], stride[1], stride[2],
        dilation[0], dilation[1], dilation[2],
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_BIAS=(bias is not None),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We don't use nn.Conv3d anymore, but we keep parameters to initialize weight/bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.use_bias = bias

        # Initialize weights and bias manually
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3d_triton(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )