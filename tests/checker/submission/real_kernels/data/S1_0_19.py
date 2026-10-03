import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pointwise_conv_kernel(
    X, W, Bias, Y,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc,
    stride_yn, stride_yc, stride_yh, stride_yw,
    H, W, C_in, C_out,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    w_start = pid_w * BLOCK_SIZE
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE)
    mask_w = w_offsets < W

    base_x = pid_n * stride_xn + pid_h * stride_xh
    base_y = pid_n * stride_yn + pid_h * stride_yh

    acc = tl.zeros((BLOCK_SIZE, C_out), dtype=tl.float32)

    for i in range(C_in):
        x_offsets = base_x + i * stride_xc + w_offsets
        x_vec = tl.load(X + x_offsets, mask=mask_w, other=0.0)

        w_offsets_col = i + tl.arange(0, C_out) * stride_wc
        w_vec = tl.load(W + w_offsets_col)

        acc += x_vec[:, None] * w_vec[None, :]

    if has_bias:
        bias_vec = tl.load(Bias + tl.arange(0, C_out))
        acc += bias_vec[None, :]

    y_offsets = base_y[:, None] + tl.arange(0, C_out)[None, :] * stride_yc + w_offsets[:, None]
    tl.store(Y + y_offsets, acc, mask=mask_w[:, None])


def triton_pointwise_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]

    out = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 64
    grid = (N, H, (W + BLOCK_SIZE - 1) // BLOCK_SIZE)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wn, stride_wc = weight.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = out.stride()

    pointwise_conv_kernel[grid](
        x, weight, bias, out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wn, stride_wc,
        stride_yn, stride_yc, stride_yh, stride_yw,
        H, W, C_in, C_out,
        has_bias=(bias is not None),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv1d.weight
        bias = self.conv1d.bias
        return triton_pointwise_conv(x, weight, bias)