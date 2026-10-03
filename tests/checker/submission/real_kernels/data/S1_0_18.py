import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    X, W, B, OUT,
    N, H, W, H_out, W_out, C,
    stride, padding,
    KH: tl.constexpr, KW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_spatial = tl.program_id(1)

    if pid_n >= N or pid_spatial >= H_out * W_out:
        return

    h_out = pid_spatial // W_out
    w_out = pid_spatial % W_out

    h_in = h_out * stride - padding
    w_in = w_out * stride - padding

    c = pid_n % C

    # Kernel coordinate ranges
    kh = tl.arange(0, KH)
    kw = tl.arange(0, KW)

    # Corresponding input coordinates
    ih = h_in + kh
    iw = w_in + kw

    # Create 2D mask for boundary checks
    mask = (ih[:, None] >= 0) & (ih[:, None] < H) & \
           (iw[None, :] >= 0) & (iw[None, :] < W)

    # Compute flat indices for input and weight
    x_idx = pid_n * H * W + ih[:, None] * W + iw[None, :]
    x_val = tl.load(X + x_idx, mask=mask, other=0.0)

    w_idx = c * KH * KW + kh[:, None] * KW + kw[None, :]
    w_val = tl.load(W + w_idx)

    # Element-wise multiply and sum reduction
    acc = tl.sum(x_val * w_val)

    # Add bias
    bias_val = tl.load(B + c)
    acc += bias_val

    # Store result
    out_idx = pid_n * H_out * W_out + h_out * W_out + w_out
    tl.store(OUT + out_idx, acc.to(tl.float32))


def triton_depthwise_conv2d(x, weight, bias, stride, padding):
    B, C, H, W = x.shape
    N = B * C
    KH, KW = weight.shape[2], weight.shape[3]
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    # Reshape to merge batch and channel dimensions for independent processing
    x_reshaped = x.reshape(N, H, W).contiguous()
    w_reshaped = weight.reshape(C, KH, KW).contiguous()
    out_reshaped = torch.empty((N, H_out, W_out), dtype=x.dtype, device=x.device)

    # Handle optional bias
    if bias is None:
        b_tensor = torch.zeros(C, dtype=x.dtype, device=x.device)
    else:
        b_tensor = bias

    # Launch kernel: one program per (batch*channel, spatial_location)
    grid = (N, H_out * W_out)
    depthwise_conv2d_kernel[grid](
        x_reshaped, w_reshaped, b_tensor, out_reshaped,
        N, H, W, H_out, W_out, C,
        stride, padding,
        KH, KW
    )

    return out_reshaped.reshape(B, C, H_out, W_out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        weight = self.conv2d.weight
        bias = self.conv2d.bias
        stride = self.conv2d.stride[0]
        padding = self.conv2d.padding[0]
        return triton_depthwise_conv2d(x, weight, bias, stride, padding)