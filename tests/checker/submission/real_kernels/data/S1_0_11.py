import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    X, W, B, Out,
    stride_x_h, stride_x_w, stride_x_c, stride_x_n,
    stride_w_h, stride_w_w, stride_w_c, stride_w_oc,
    stride_out_h, stride_out_w, stride_out_c, stride_out_n,
    N, OC, IC, H, W,
    KH, KW,
    SH, SW,
    PH, PW,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Offsets for the output block
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_oc = pid_oc * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
    off_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    off_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Masks to prevent out-of-bounds access
    mask_n = off_n < N
    mask_oc = off_oc < OC
    mask_h = off_h < H
    mask_w = off_w < W

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_OC, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Loop over Input Channels (IC) and Kernel dimensions (KH, KW)
    for ic in range(0, IC):
        for kh in range(0, KH):
            for kw in range(0, KW):
                # Source coordinates in X
                src_h = off_h * SH + kh - PH
                src_w = off_w * SW + kw - PW

                # Load X
                x_ptr = X + off_n[:, None, None, None] * stride_x_n + \
                            off_oc[None, :, None, None] * stride_x_c + \
                            src_h[None, None, :, None] * stride_x_h + \
                            src_w[None, None, None, :] * stride_x_w

                # Load W
                w_ptr = W + off_oc[:, None, None, None] * stride_w_oc + \
                            ic[None, :, None, None] * stride_w_c + \
                            kh[None, None, :, None] * stride_w_h + \
                            kw[None, None, None, :] * stride_w_w

                # Load values with masking
                x_mask = (mask_n[:, None, None, None] & 
                          mask_oc[None, :, None, None] & 
                          (src_h[None, None, :, None] >= 0) & (src_h[None, None, :, None] < H) & 
                          (src_w[None, None, None, :] >= 0) & (src_w[None, None, None, :] < W))

                x_val = tl.load(x_ptr, mask=x_mask, other=0.0)
                w_val = tl.load(w_ptr, mask=mask_oc[:, None, None, None], other=0.0)

                acc += x_val * w_val

    # Add Bias if provided
    if B is not None:
        b_ptr = B + off_oc
        b_val = tl.load(b_ptr, mask=mask_oc, other=0.0)
        acc += b_val[None, :, None, None]

    # Apply ReLU activation
    acc = tl.maximum(acc, 0.0)

    # Store result
    out_ptr = Out + off_n[:, None, None, None] * stride_out_n + \
                    off_oc[None, :, None, None] * stride_out_c + \
                    off_h[None, None, :, None] * stride_out_h + \
                    off_w[None, None, None, :] * stride_out_w

    tl.store(out_ptr, acc, mask=(mask_n[:, None, None, None] & 
                                 mask_oc[None, :, None, None] & 
                                 mask_h[None, None, :, None] & 
                                 mask_w[None, None, None, :]))


def triton_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert groups == 1 and dilation == 1, "Only groups=1 and dilation=1 supported."

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, IC, H, W = x.shape
    OC, IC_W, KH, KW = weight.shape

    assert IC == IC_W, "Input channels must match weight input channels"

    SH, SW = stride, stride
    PH, PW = padding, padding

    # Output dimensions
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    out = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_N = 1
    BLOCK_SIZE_OC = 1
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    grid = (
        N,
        OC,
        (OH + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (OW + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )

    conv2d_kernel[grid](
        x, weight, bias, out,
        x.stride(3), x.stride(2), x.stride(1), x.stride(0),
        weight.stride(3), weight.stride(2), weight.stride(1), weight.stride(0),
        out.stride(3), out.stride(2), out.stride(1), out.stride(0),
        N, OC, IC, H, W,
        KH, KW,
        SH, SW,
        PH, PW,
        BLOCK_SIZE_N, BLOCK_SIZE_OC, BLOCK_SIZE_H, BLOCK_SIZE_W
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized Model using custom Triton kernels for 2D Convolution + ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the parameters as in the original model
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the custom Triton kernel.
        """
        return triton_conv2d(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            stride=self.conv2d.stride[0],
            padding=self.conv2d.padding[0],
            dilation=self.conv2d.dilation[0],
            groups=self.conv2d.groups
        )