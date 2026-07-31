import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    X, W, Bias, Out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wh, stride_wk,
    stride_on, stride_oc, stride_oh, stride_ow,
    stride_bc,
    C, H, W, Kh, Kw,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    H_out, W_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    USE_BIAS: tl.constexpr
):
    # Program IDs
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Extract N and C from the flattened NC index
    n = pid_nc // C
    c = pid_nc % C

    # Output tile offsets
    offs_h = pid_h * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_w = pid_w * BLOCK_N + tl.arange(0, BLOCK_N)

    # Output boundary mask
    mask_h = offs_h < H_out
    mask_w = offs_w < W_out
    mask_out = mask_h[:, None] & mask_w[None, :]

    # Accumulator initialized to zero
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over kernel height and width
    for kh in range(Kh):
        for kw in range(Kw):
            # Calculate input coordinates for the current kernel element
            # Broadcasting: offs_h is (BLOCK_M,), offs_w is (BLOCK_N,)
            # in_h becomes (BLOCK_M, 1), in_w becomes (1, BLOCK_N)
            in_h = offs_h[:, None] * stride_h + kh * dilation_h - pad_h
            in_w = offs_w[None, :] * stride_w + kw * dilation_w - pad_w

            # Input boundary mask
            mask_in = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)

            # Load input tile
            # X shape: [N, C, H, W]
            x_ptr = X + n * stride_xn + c * stride_xc
            x = tl.load(x_ptr + in_h * stride_xh + in_w * stride_xw, mask=mask_in, other=0.0)

            # Load kernel weight
            # W shape: [C, Kh, Kw]
            w_ptr = W + c * stride_wc + kh * stride_wh + kw * stride_wk
            w_val = tl.load(w_ptr)

            # Accumulate
            acc += x * w_val

    # Add bias if present
    if USE_BIAS:
        bias_ptr = Bias + c * stride_bc
        bias_val = tl.load(bias_ptr)
        acc += bias_val

    # Store output
    out_ptr = Out + n * stride_on + c * stride_oc
    tl.store(out_ptr + offs_h[:, None] * stride_oh + offs_w[None, :] * stride_ow, acc, mask=mask_out)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias

        # Register weights and bias
        # Original model uses nn.Conv2d(in_channels, in_channels, ..., groups=in_channels)
        # So weight shape is [in_channels, 1, Kh, Kw]
        self.register_buffer('weight', torch.randn(in_channels, in_channels // groups, kernel_size_h, kernel_size_w))
        if bias:
            self.register_buffer('bias', torch.zeros(in_channels))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        Kh = self.kernel_size_h
        Kw = self.kernel_size_w
        Sh = self.stride_h
        Sw = self.stride_w
        Ph = self.padding_h
        Pw = self.padding_w
        Dh = self.dilation_h
        Dw = self.dilation_w

        # Calculate output dimensions
        H_out = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
        W_out = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1

        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        # Prepare weight tensor for kernel: [C, Kh, Kw]
        w = self.weight.squeeze(1)

        # Define block sizes
        BLOCK_M = 16
        BLOCK_N = 16

        # Grid configuration: (N*C, num_h_tiles, num_w_tiles)
        grid = (
            N * C,
            (H_out + BLOCK_M - 1) // BLOCK_M,
            (W_out + BLOCK_N - 1) // BLOCK_N
        )

        # Launch kernel
        depthwise_conv2d_kernel[grid](
            x, w, self.bias, out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            self.bias.stride(0) if self.bias is not None else 0,
            C, H, W, Kh, Kw,
            Sh, Sw, Ph, Pw, Dh, Dw,
            H_out, W_out,
            BLOCK_M, BLOCK_N,
            USE_BIAS=self.bias is not None
        )

        return out