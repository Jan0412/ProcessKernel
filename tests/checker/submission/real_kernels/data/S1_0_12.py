import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    X,
    W,
    Bias,
    Out,
    stride_xh, stride_xw,
    stride_oh, stride_ow,
    stride_wh, stride_wk,
    H, W,
    KH, KW,
    PH, PW,
    stride_h, stride_w,
    dilation_h, dilation_w,
    n_channels,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KH_T: tl.constexpr,
    KW_T: tl.constexpr,
):
    # Program IDs
    pid_bc = tl.program_id(0)  # batch * channel
    pid_spatial = tl.program_id(1)  # flattened spatial index

    # Determine output spatial coordinates (h, w) for this block
    # We layout the spatial grid as (height * width)
    # But to make tiling easy, we can view it as a 2D grid of blocks
    # However, Triton grid is 1D here (or we can use 2D). Let's use 2D grid for spatial:
    # Grid X: (batch * channel)
    # Grid Y: (ceil(H / BLOCK_M), ceil(W / BLOCK_N))
    # Wait, the signature above implies pid_spatial is a flat index if we want to keep it simple,
    # but 2D grid is better for mapping to H, W.

    # Let's restructure the grid launch in the wrapper to be 3D:
    # Grid = (batch, channels, num_spatial_blocks)
    # But Triton grid is usually 1D, 2D, or 3D.
    # Let's stick to a 2D grid: pid_bc (batch * channel) and pid_spatial (flattened spatial blocks)
    # Actually, let's use a 3D grid: (grid_b, grid_c, grid_spatial)
    # To simplify kernel logic, let's use a 2D grid:
    # pid_bc covers (batch * channels)
    # pid_spatial covers (height // BLOCK_M * width // BLOCK_N)

    # Calculate H, W start coordinates for this block
    # We need to know the dimensions to map pid_spatial to (h_start, w_start)
    # Let's pass H and W to the kernel and compute num_blocks_w
    num_blocks_w = tl.cdiv(W, BLOCK_N)

    block_idx = pid_spatial
    w_start = (block_idx % num_blocks_w) * BLOCK_N
    h_start = (block_idx // num_blocks_w) * BLOCK_M

    # Offsets within the block
    offs_m = h_start + tl.arange(0, BLOCK_M)
    offs_n = w_start + tl.arange(0, BLOCK_N)

    # Create 2D masks for spatial bounds
    mask_m = offs_m < H
    mask_n = offs_n < W

    # Outer product mask for the output block
    mask_out = mask_m[:, None] & mask_n[None, :]

    # Load kernel into shared memory / registers
    # Kernel shape: (KH, KW)
    # We can load the kernel once per program
    # Offsets for kernel
    offs_kh = tl.arange(0, KH_T)
    offs_kw = tl.arange(0, KW_T)

    # Create kernel mask for dilation/padding logic if needed, 
    # but simpler to just load with mask and multiply by 0 if out of bounds
    # The input window corresponds to:
    # h_input = h_out * stride_h - PH + kh * dilation_h
    # w_input = w_out * stride_w - PW + kw * dilation_w

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kernel dimensions
    for kh in range(KH_T):
        for kw in range(KW_T):
            # Calculate input coordinates
            # h_in = h_out * stride_h - PH + kh * dilation_h
            # w_in = w_out * stride_w - PW + kw * dilation_w

            # Vectorized calculation for the block
            # Shape (BLOCK_M, BLOCK_N)
            h_in = offs_m * stride_h - PH + kh * dilation_h
            w_in = offs_n * stride_w - PW + kw * dilation_w

            # Calculate input pointer
            # Base pointer for this batch/channel
            x_ptr = X + pid_bc * stride_xh * H * stride_xw * W # This is wrong strides

            # Correct strides:
            # X shape: (B, C, H, W)
            # We are at batch, channel `pid_bc`.
            # We need to calculate the offset for each (h_in, w_in)

            # Let's load the kernel value
            w_val = tl.load(W + kh * stride_wh + kw * stride_wk, mask=True, other=0.0)

            # Calculate input indices
            # Mask for valid input indices
            mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)

            # Calculate linear offset for input
            # offset = h_in * stride_h_in + w_in * stride_w_in
            # stride_h_in is usually W, stride_w_in is 1
            x_offset = h_in * stride_xw + w_in # Assuming stride_xw is the width stride (1) and we passed W as stride_xh?

            # Let's clarify strides passed:
            # stride_xh: stride for height dimension in X (usually W)
            # stride_xw: stride for width dimension in X (usually 1)

            x_ptr_curr = X + pid_bc * stride_xh * H + x_offset

            # Load input
            x_val = tl.load(x_ptr_curr, mask=mask_in & mask_out, other=0.0)

            # Accumulate
            acc += x_val * w_val

    # Add bias if present
    if Bias is not None:
        bias_val = tl.load(Bias + pid_bc)
        acc = acc + bias_val

    # Store output
    out_ptr = Out + pid_bc * stride_oh * H + \
              offs_m[:, None] * stride_ow + \
              offs_n[None, :]

    tl.store(out_ptr, acc, mask=mask_out)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Custom Triton kernel for depthwise convolution.
    x: (B, C, H, W)
    weight: (C, 1, KH, KW) - Since groups=C, each channel has its own kernel
    """
    B, C, H, W = x.shape
    KH, KW = weight.shape[2], weight.shape[3]

    # Calculate output dimensions
    OH = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1

    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)

    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()

    # Grid configuration
    # We launch a program for each (Batch, Channel) and each spatial block
    BLOCK_M, BLOCK_N = 4, 4
    grid_spatial = (triton.cdiv(OH, BLOCK_M), triton.cdiv(OW, BLOCK_N))
    grid_bc = B * C

    # Total grid size: (grid_bc, grid_spatial[0], grid_spatial[1])
    # Triton grid is usually 1D, 2D, or 3D. Let's use 3D grid.
    grid = (grid_bc, grid_spatial[0], grid_spatial[1])

    # Strides
    # X: (B, C, H, W) -> strides: (C*H*W, H*W, W, 1)
    # But we are indexing by pid_bc (B*C) and spatial.
    # So effectively we treat X as (B*C, H, W)

    depthwise_conv2d_kernel[grid](
        x,
        weight.view(C, KH, KW), # Flatten groups
        bias,
        out,
        H * W, 1, # stride_xh, stride_xw
        OH * OW, 1, # stride_oh, stride_ow
        KW, 1, # stride_wh, stride_wk
        OH, OW,
        KH, KW,
        padding, padding,
        stride, stride,
        dilation, dilation,
        C,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        KH_T=KH,
        KW_T=KW,
    )
    return out


@triton.jit
def pointwise_conv2d_kernel(
    X,
    W,
    Bias,
    Out,
    stride_xc,
    stride_wc,
    stride_oc,
    C_IN,
    C_OUT,
    BLOCK_C: tl.constexpr,
):
    # pid_spatial: iterates over (batch * height * width)
    # pid_c: iterates over chunks of out_channels
    pid_spatial = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Offsets for output channels
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C_OUT

    # Load input vector for this spatial location
    # X shape: (B, C_IN, H, W)
    # We treat it as (B*H*W, C_IN)
    x_ptr = X + pid_spatial * stride_xc + tl.arange(0, C_IN)
    x = tl.load(x_ptr, mask=True, other=0.0)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels
    for i in range(C_IN):
        # Load kernel weights for this input channel i and output channels offs_c
        # Weight shape: (C_OUT, C_IN, 1, 1) -> (C_OUT, C_IN)
        # We want W[offs_c, i]
        w_ptr = W + offs_c * stride_wc + i
        w = tl.load(w_ptr, mask=mask_c, other=0.0)
        acc += x[i] * w

    # Add bias
    if Bias is not None:
        bias_ptr = Bias + offs_c
        bias = tl.load(bias_ptr, mask=mask_c, other=0.0)
        acc = acc + bias

    # Store output
    out_ptr = Out + pid_spatial * stride_oc + offs_c
    tl.store(out_ptr, acc, mask=mask_c)


def triton_pointwise_conv2d(x, weight, bias=None):
    """
    Custom Triton kernel for pointwise convolution (1x1 conv).
    x: (B, C_IN, H, W)
    weight: (C_OUT, C_IN, 1, 1)
    """
    B, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]

    out = torch.empty((B, C_OUT, H, W), device=x.device, dtype=x.dtype)

    x = x.contiguous()
    weight = weight.contiguous()

    # Grid configuration
    # Spatial dimension: B * H * W
    # Channel dimension: chunks of BLOCK_C
    BLOCK_C = 64
    grid_spatial = B * H * W
    grid_c = triton.cdiv(C_OUT, BLOCK_C)

    grid = (grid_spatial, grid_c)

    # Strides
    # X: (B, C_IN, H, W) -> stride for C_IN is H*W
    # W: (C_OUT, C_IN) -> stride for C_OUT is C_IN
    # Out: (B, C_OUT, H, W) -> stride for C_OUT is H*W

    pointwise_conv2d_kernel[grid](
        x,
        weight.view(C_OUT, C_IN),
        bias,
        out,
        H * W, # stride_xc
        C_IN, # stride_wc
        H * W, # stride_oc
        C_IN,
        C_OUT,
        BLOCK_C=BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Depthwise-Separable 2D Convolution using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weights
        # Depthwise: (in_channels, 1, kernel_size, kernel_size)
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.depthwise_bias = nn.Parameter(torch.zeros(in_channels)) if bias else None

        # Pointwise: (out_channels, in_channels, 1, 1)
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        self.pointwise_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise Convolution
        x = triton_depthwise_conv2d(
            x, 
            self.depthwise_weight, 
            self.depthwise_bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )

        # Pointwise Convolution
        x = triton_pointwise_conv2d(
            x, 
            self.pointwise_weight, 
            self.pointwise_bias
        )

        return x