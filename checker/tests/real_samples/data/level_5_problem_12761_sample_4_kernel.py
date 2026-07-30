import torch
import torchvision
import triton
import triton.language as tl
from torch import nn
import torch.nn.functional as F


# --------------------------- Triton kernels ---------------------------

@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    """Element‑wise ReLU implemented in Triton."""
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def maxpool2d_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)

    # total output elements = N*C*H_out*W_out, where H_out = H//stride, W_out = W//stride
    H_out = H // stride
    W_out = W // stride
    total = N * C * H_out * W_out
    mask = offs < total

    # linear index -> (n, c, ho, wo)
    n = offs // (C * H_out * W_out)
    rem = offs % (C * H_out * W_out)
    c = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    ho = rem2 // W_out
    wo = rem2 % W_out

    # input coordinates
    hi = ho * stride
    wi = wo * stride

    # load the 2×2 window
    def load(i_off, j_off):
        ptr = x_ptr + ((n * C + c) * H + (hi + i_off)) * W + (wi + j_off)
        return tl.load(ptr, mask=mask, other=-float('inf'))

    v00 = load(0, 0)
    v01 = load(0, 1)
    v10 = load(1, 0)
    v11 = load(1, 1)

    out_val = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    tl.store(out_ptr + offs, out_val, mask=mask)


def triton_maxpool2d(x: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """2×2 max‑pool with given stride (default 2)."""
    assert x.is_cuda
    N, C, H, W = x.shape
    assert H % stride == 0 and W % stride == 0, "Spatial dimensions must be divisible by stride"
    out = torch.empty((N, C, H // stride, W // stride), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 128
    total_out = out.numel()
    grid = lambda meta: ((total_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    maxpool2d_kernel[grid](
        x,
        out,
        N,
        C,
        H,
        W,
        stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def upsample_nearest_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    scale,
    out_H,
    out_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)

    total = N * C * out_H * out_W
    mask = offs < total

    n = offs // (C * out_H * out_W)
    rem = offs % (C * out_H * out_W)
    c = rem // (out_H * out_W)
    rem2 = rem % (out_H * out_W)
    ho = rem2 // out_W
    wo = rem2 % out_W

    hi = ho // scale
    wi = wo // scale

    src_ptr = x_ptr + ((n * C + c) * H + hi) * W + wi
    val = tl.load(src_ptr, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


def triton_upsample_nearest(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Nearest‑neighbor up‑sample to a given (H, W). Assumes integer scaling."""
    assert x.is_cuda
    N, C, H, W = x.shape
    out_H, out_W = size
    scale_h = out_H // H
    scale_w = out_W // W
    assert scale_h == scale_w, "Only uniform integer scaling supported"
    scale = scale_h
    out = torch.empty((N, C, out_H, out_W), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128
    total = out.numel()
    grid = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    upsample_nearest_kernel[grid](
        x,
        out,
        N,
        C,
        H,
        W,
        scale,
        out_H,
        out_W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# --------------------------- Model definition ---------------------------

class Block(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, batch_norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channel, out_channels, kernel_size=3, padding=1)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(mid_channel)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = triton_relu(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = triton_relu(x)
        return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode='nearest'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode

        # initial 1×1 conv
        self.init_conv = nn.Conv2d(in_channels, 3, kernel_size=1)

        # VGG‑11 pretrained encoder
        encoder = torchvision.models.vgg11(pretrained=True).features
        self.conv1 = encoder[0]
        self.conv2 = encoder[3]
        self.conv3 = encoder[6]
        self.conv3s = encoder[8]
        self.conv4 = encoder[11]
        self.conv4s = encoder[13]
        self.conv5 = encoder[16]
        self.conv5s = encoder[18]

        # decoder blocks
        self.center = Block(512, 512, 256, batch_norm)
        self.dec5 = Block(512 + 256, 512, 256, batch_norm)
        self.dec4 = Block(512 + 256, 512, 128, batch_norm)
        self.dec3 = Block(256 + 128, 256, 64, batch_norm)
        self.dec2 = Block(128 + 64, 128, 32, batch_norm)
        self.dec1 = Block(64 + 32, 64, 32, batch_norm)

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    # replace torch.nn.functional.interpolate
    def up(self, x, size):
        return triton_upsample_nearest(x, size)

    # replace torch.nn.MaxPool2d
    def down(self, x):
        return triton_maxpool2d(x, stride=2)

    def forward(self, x):
        init_conv = triton_relu(self.init_conv(x))

        enc1 = triton_relu(self.conv1(init_conv))
        enc2 = triton_relu(self.conv2(self.down(enc1)))
        enc3 = triton_relu(self.conv3(self.down(enc2)))
        enc3 = triton_relu(self.conv3s(enc3))
        enc4 = triton_relu(self.conv4(self.down(enc3)))
        enc4 = triton_relu(self.conv4s(enc4))
        enc5 = triton_relu(self.conv5(self.down(enc4)))
        enc5 = triton_relu(self.conv5s(enc5))

        center = self.center(self.down(enc5))

        dec5 = self.dec5(torch.cat([self.up(center, enc5.shape[-2:]), enc5], dim=1))
        dec4 = self.dec4(torch.cat([self.up(dec5, enc4.shape[-2:]), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.shape[-2:]), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.shape[-2:]), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.shape[-2:]), enc1], dim=1))

        out = self.out(dec1)
        return out