import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import triton
import triton.language as tl


# -------------------------------------------------
# Triton kernels
# -------------------------------------------------
@triton.jit
def pixel_norm_reduce_kernel(
    x_ptr, sigma_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * H * W
    if pid >= total:
        return

    # (n, h, w) coordinates for this program
    n = pid // (H * W)
    rem = pid % (H * W)
    h = rem // W
    w = rem % W

    acc = tl.zeros([1], dtype=tl.float32)

    # reduction over channel dimension
    for off in range(0, C, BLOCK_SIZE):
        cur = tl.arange(0, BLOCK_SIZE) + off
        mask = cur < C
        x = tl.load(
            x_ptr + n * stride_n + cur * stride_c + h * stride_h + w * stride_w,
            mask=mask,
            other=0.0,
        )
        acc += tl.sum(x * x, axis=0)

    sigma = tl.sqrt(acc + 1e-5)
    # sigma has shape (N, 1, H, W) – channel dimension is omitted
    tl.store(sigma_ptr + n * stride_n + h * stride_h + w * stride_w, sigma)


@triton.jit
def pixel_norm_divide_kernel(
    x_ptr,
    sigma_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    sigma_stride_n,
    sigma_stride_h,
    sigma_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * H * W
    if pid >= total:
        return

    n = pid // (C * H * W)
    rem = pid % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W

    x = tl.load(
        x_ptr + n * stride_n + c * stride_c + h * stride_h + w * stride_w
    )
    sigma = tl.load(
        sigma_ptr + n * sigma_stride_n + h * sigma_stride_h + w * sigma_stride_w
    )
    out = x / sigma
    tl.store(
        out_ptr + n * stride_n + c * stride_c + h * stride_h + w * stride_w, out
    )


@triton.jit
def add_scaled_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    factor,
    N,
    C,
    H,
    W,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * H * W
    if pid >= total:
        return

    n = pid // (C * H * W)
    rem = pid % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W

    a = tl.load(
        a_ptr + n * stride_n + c * stride_c + h * stride_h + w * stride_w
    )
    b = tl.load(
        b_ptr + n * stride_n + c * stride_c + h * stride_h + w * stride_w
    )
    out = a + factor * b
    tl.store(
        out_ptr + n * stride_n + c * stride_c + h * stride_h + w * stride_w, out
    )


# -------------------------------------------------
# Helper wrappers
# -------------------------------------------------
def triton_pixel_norm(x: torch.Tensor) -> torch.Tensor:
    """Pixel‑wise normalization using two Triton kernels."""
    assert x.is_cuda, "Triton pixel_norm requires CUDA tensor"
    N, C, H, W = x.shape
    stride_n, stride_c, stride_h, stride_w = x.stride()

    # allocate sigma (N,1,H,W)
    sigma = torch.empty((N, 1, H, W), dtype=x.dtype, device=x.device)

    # reduction kernel
    BLOCK_C = 64
    grid_reduce = lambda meta: ((N * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    pixel_norm_reduce_kernel[grid_reduce](
        x,
        sigma,
        N,
        C,
        H,
        W,
        stride_n,
        stride_c,
        stride_h,
        stride_w,
        BLOCK_SIZE=BLOCK_C,
    )

    # division kernel
    out = torch.empty_like(x)
    total = N * C * H * W
    BLOCK = 128
    grid_div = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    pixel_norm_divide_kernel[grid_div](
        x,
        sigma,
        out,
        N,
        C,
        H,
        W,
        stride_n,
        stride_c,
        stride_h,
        stride_w,
        sigma.stride(0),
        sigma.stride(2),
        sigma.stride(3),
        BLOCK_SIZE=BLOCK,
    )
    return out


def triton_add_scaled(a: torch.Tensor, b: torch.Tensor, factor: float) -> torch.Tensor:
    """Compute a + factor * b with a fused Triton kernel."""
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()
    N, C, H, W = a.shape
    stride_n, stride_c, stride_h, stride_w = a.stride()

    out = torch.empty_like(a)

    total = N * C * H * W
    BLOCK = 128
    grid = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_scaled_kernel[grid](
        a,
        b,
        out,
        factor,
        N,
        C,
        H,
        W,
        stride_n,
        stride_c,
        stride_h,
        stride_w,
        BLOCK_SIZE=BLOCK,
    )
    return out


# -------------------------------------------------
# Original utilities (kept unchanged)
# -------------------------------------------------
def pixel_norm(x):
    sigma = x.norm(dim=1, keepdim=True)
    out = x / (sigma + 1e-05)
    return out


class EqualizedLR(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self._make_params()

    def _make_params(self):
        weight = self.module.weight
        height = weight.data.shape[0]
        width = weight.view(height, -1).data.shape[1]
        del self.module._parameters["weight"]
        self.module.weight = None
        self.weight = nn.Parameter(weight.data)
        self.factor = np.sqrt(2 / width)
        nn.init.normal_(self.weight)
        self.bias = self.module.bias
        self.module.bias = None
        if self.bias is not None:
            del self.module._parameters["bias"]
            nn.init.zeros_(self.bias)

    def forward(self, *args, **kwargs):
        self.module.weight = self.factor * self.weight
        if self.bias is not None:
            self.module.bias = 1.0 * self.bias
        out = self.module.forward(*args, **kwargs)
        self.module.weight = None
        self.module.bias = None
        return out


# -------------------------------------------------
# Optimized model
# -------------------------------------------------
class ModelNew(nn.Module):
    def __init__(
        self,
        f_in,
        f_out=None,
        f_hidden=None,
        is_bias=True,
        actvn=F.relu,
        factor=1.0,
        eq_lr=False,
        pixel_norm=False,
    ):
        super().__init__()
        if f_out is None:
            f_out = f_in
        if f_hidden is None:
            f_hidden = min(f_in, f_out)
        self.f_in = f_in
        self.f_hidden = f_hidden
        self.f_out = f_out
        self.factor = factor
        self.eq_lr = eq_lr
        self.use_pixel_norm = pixel_norm
        self.actvn = actvn
        self.conv_0 = nn.Conv2d(self.f_in, self.f_hidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            self.f_hidden, self.f_out, 3, stride=1, padding=1, bias=is_bias
        )
        if self.eq_lr:
            self.conv_0 = EqualizedLR(self.conv_0)
            self.conv_1 = EqualizedLR(self.conv_1)
        if f_in == f_out:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Conv2d(f_in, f_out, 1, bias=False)
            if self.eq_lr:
                self.shortcut = EqualizedLR(self.shortcut)
        nn.init.zeros_(self.conv_1.weight)

    def forward(self, x):
        x_s = self.shortcut(x)
        if self.use_pixel_norm:
            x = triton_pixel_norm(x)
        dx = self.conv_0(self.actvn(x))
        if self.use_pixel_norm:
            dx = triton_pixel_norm(dx)
        dx = self.conv_1(self.actvn(dx))
        # fused addition + scaling
        out = triton_add_scaled(x_s, dx, self.factor)
        return out