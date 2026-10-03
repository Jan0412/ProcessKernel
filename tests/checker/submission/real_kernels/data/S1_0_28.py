import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_relu_hardswish_kernel(
    X, W, B, Y,
    N, C_IN, C_OUT, H_IN, W_IN, H_OUT, W_OUT, K,
    PAD, STRIDE,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    K: tl.constexpr,
):
    # Each program handles one (N, out_h, out_w) position and computes all C_OUT channels
    pid = tl.program_id(0)
    n = pid // (H_OUT * W_OUT)
    rem = pid % (H_OUT * W_OUT)
    out_h = rem // W_OUT
    out_w = rem % W_OUT

    # Strides for input X (N, C_IN, H_IN, W_IN)
    stride_x_n = C_IN * H_IN * W_IN
    stride_x_c = H_IN * W_IN
    stride_x_h = W_IN
    stride_x_w = 1

    # Strides for weight W (C_OUT, C_IN, K, K)
    stride_w_c = C_IN * K * K
    stride_w_kh = K
    stride_w_kw = 1

    # Strides for output Y (N, C_OUT, H_OUT, W_OUT)
    stride_y_n = C_OUT * H_OUT * W_OUT
    stride_y_c = H_OUT * W_OUT
    stride_y_h = W_OUT
    stride_y_w = 1

    # Base pointer for output at (n, 0, out_h, out_w)
    y_ptr = Y + n * stride_y_n + out_h * stride_y_h + out_w * stride_y_w

    # Channel offsets for vectorized loads
    c_offsets = tl.arange(0, C_IN)

    for c in range(C_OUT):
        acc = tl.load(B + c)

        # Convolution accumulation over kernel and input channels
        for kh in range(K):
            for kw in range(K):
                in_h = out_h + kh * STRIDE - PAD
                in_w = out_w + kw * STRIDE - PAD

                # Boundary mask
                mask = (in_h >= 0) & (in_h < H_IN) & (in_w >= 0) & (in_w < W_IN)

                # Load input slice: X[n, :, in_h, in_w]
                x_ptr = X + n * stride_x_n + in_h * stride_x_h + in_w * stride_x_w
                x_vals = tl.load(x_ptr + c_offsets * stride_x_c, mask=mask, other=0.0)

                # Load weight slice: W[c, :, kh, kw]
                w_ptr = W + c * stride_w_c + kh * stride_w_kh + kw * stride_w_kw
                w_vals = tl.load(w_ptr + c_offsets, mask=mask, other=0.0)

                acc += tl.sum(x_vals * w_vals)

        # Fused Activation: ReLU * HardSwish
        # ReLU part
        relu_x = tl.maximum(acc, 0.0)
        # HardSwish part: clamp((x + 3) / 6, 0, 1)
        hs_input = (acc + 3.0) / 6.0
        hs = tl.minimum(tl.maximum(hs_input, 0.0), 1.0)

        out_val = relu_x * hs
        tl.store(y_ptr + c * stride_y_c, out_val)


def triton_conv2d_relu_hardswish(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                                  stride: int = 1, padding: int = 1):
    """
    Fused Conv2d + ReLU + HardSwish kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    else:
        bias = torch.zeros(weight.shape[0], dtype=x.dtype, device=x.device)

    N, C_IN, H_IN, W_IN = x.shape
    C_OUT, _, K, _ = weight.shape
    H_OUT = (H_IN + 2 * padding - K) // stride + 1
    W_OUT = (W_IN + 2 * padding - K) // stride + 1

    y = torch.empty((N, C_OUT, H_OUT, W_OUT), dtype=x.dtype, device=x.device)

    grid = (N * H_OUT * W_OUT,)
    conv2d_relu_hardswish_kernel[grid](
        x, weight, bias, y,
        N, C_IN, C_OUT, H_IN, W_IN, H_OUT, W_OUT, K,
        padding, stride,
        C_IN=C_IN, C_OUT=C_OUT, K=K
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, then applies a fused ReLU + HardSwish activation.
    Uses a custom Triton kernel to fuse the convolution and activation into a single pass.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # Keep the standard module solely as a weight/bias holder
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # Extract parameters and launch the fused Triton kernel
        # Default stride=1, padding=kernel_size//2 for kernel_size=3
        padding = self.conv.kernel_size[0] // 2
        stride = self.conv.stride[0]
        return triton_conv2d_relu_hardswish(
            x, self.conv.weight, self.conv.bias, stride=stride, padding=padding
        )