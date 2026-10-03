import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    X, W, O,
    B, C_in, C_out, H, W, H_out, W_out, K,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    pid_b = pid_bc // C_out
    pid_c = pid_bc % C_out

    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W

    oh_offsets = oh_start + tl.arange(0, BLOCK_H)
    ow_offsets = ow_start + tl.arange(0, BLOCK_W)

    oh_2d = oh_offsets[:, None]
    ow_2d = ow_offsets[None, :]
    mask = (oh_2d < H_out) & (ow_2d < W_out)

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    x_base = X + pid_b * C_in * H * W
    w_base = W + pid_c * C_in * K * K
    o_base = O + pid_b * C_out * H_out * W_out + pid_c * H_out * W_out

    for c_in_idx in range(C_in):
        x_tile_base = x_base + c_in_idx * H * W
        w_tile_base = w_base + c_in_idx * K * K

        for kh in range(K):
            for kw in range(K):
                ih = oh_2d - kh
                iw = ow_2d - kw
                valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                x_val = tl.load(x_tile_base + ih * W + iw, mask=valid, other=0.0)
                w_val = tl.load(w_tile_base + kh * K + kw)
                acc += x_val * w_val

    tl.store(o_base + oh_2d * W_out + ow_2d, acc, mask=mask)

def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor):
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()

    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape
    H_out = H + K - 1
    W_out = W + K - 1

    out = torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

    BLOCK_H = 16
    BLOCK_W = 16

    num_tiles_h = (H_out + BLOCK_H - 1) // BLOCK_H
    num_tiles_w = (W_out + BLOCK_W - 1) // BLOCK_W

    grid = (B * C_out, num_tiles_h, num_tiles_w)

    conv_transpose2d_kernel[grid](
        x, weight, out,
        B, C_in, C_out, H, W, H_out, W_out, K,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=4
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(x, self.conv_transpose2d.weight)