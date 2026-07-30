import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------
# Triton GEMM kernel (simple tiled matrix multiply)
# -------------------------------------------------
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # -------------------------------------------------
    # Compute block coordinates
    # -------------------------------------------------
    block_pid_m = pid // GROUP_M
    block_pid_n = pid % GROUP_M
    pid_m = block_pid_m * BLOCK_M
    pid_n = block_pid_n * BLOCK_N

    # -------------------------------------------------
    # Pointers for A and B
    # -------------------------------------------------
    a_ptrs = a_ptr + (pid_m + tl.arange(0, BLOCK_M)[:, None]) * stride_am + (tl.arange(0, BLOCK_K)[None, :] * stride_ak)
    b_ptrs = b_ptr + (tl.arange(0, BLOCK_K)[:, None] * stride_bk) + (pid_n + tl.arange(0, BLOCK_N)[None, :] * stride_bn)

    # -------------------------------------------------
    # Accumulator
    # -------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------
    # Loop over K dimension
    # -------------------------------------------------
    for k in range(0, K, BLOCK_K):
        cur_k = min(BLOCK_K, K - k)

        a = tl.load(a_ptrs + k * stride_ak, mask=(tl.arange(0, BLOCK_M)[:, None] < M - pid_m) &
                                                   (tl.arange(0, cur_k)[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs + k * stride_bk, mask=(tl.arange(0, cur_k)[:, None] < K - k) &
                                                   (tl.arange(0, BLOCK_N)[None, :] < N - pid_n), other=0.0)

        acc += tl.dot(a, b)

    # -------------------------------------------------
    # Write back the result
    # -------------------------------------------------
    c_ptrs = c_ptr + (pid_m + tl.arange(0, BLOCK_M)[:, None]) * stride_cm + (pid_n + tl.arange(0, BLOCK_N)[None, :] * stride_cn)
    mask = (tl.arange(0, BLOCK_M)[:, None] < M - pid_m) & (tl.arange(0, BLOCK_N)[None, :] < N - pid_n)
    tl.store(c_ptrs, acc, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Simple GEMM using Triton.
    a: (M, K) float32 contiguous tensor
    b: (K, N) float32 contiguous tensor
    returns c: (M, N)
    """
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tuning parameters (can be adjusted)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    total_blocks = grid_m * grid_n

    matmul_kernel[total_blocks](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    return c


# -------------------------------------------------
# Original building blocks (unchanged)
# -------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=0,
                 norm_layer=None, bias=True, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks,
                              stride=stride, padding=padding, bias=bias)
        self.norm_layer = norm_layer
        if norm_layer is not None:
            self.bn = norm_layer(out_chan, activation='leaky_relu')
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


# -------------------------------------------------
# Optimized model using Triton for the 1x1 conv
# -------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, norm_layer=None, *args, **kwargs):
        super(ModelNew, self).__init__()
        self.norm_layer = norm_layer
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1,
                               norm_layer=norm_layer)
        # 1x1 conv weight (no bias)
        self.conv_out_weight = nn.Parameter(
            torch.empty((n_classes, mid_chan), dtype=torch.float32)
        )
        self.init_weight()

    def forward(self, x):
        # x: (N, C_in, H, W)
        x = self.conv(x)                 # -> (N, mid_chan, H, W)

        N, C_mid, H, W = x.shape
        # reshape to (M, K) where M = N*H*W, K = C_mid
        x_flat = x.permute(0, 2, 3, 1).reshape(N * H * W, C_mid)

        # weight is (out_chan, in_chan); we need (in_chan, out_chan) for GEMM
        w = self.conv_out_weight.t()    # (C_mid, n_classes)

        # Triton GEMM
        out_flat = triton_matmul(x_flat, w)   # (M, n_classes)

        # reshape back to (N, n_classes, H, W)
        out = out_flat.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        return out

    def init_weight(self):
        # Initialize the 1x1 conv weight like kaiming_normal_
        nn.init.kaiming_normal_(self.conv_out_weight, a=1)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, self.norm_layer):
                nowd_params += list(module.parameters())
        # Include the custom 1x1 weight as weight decay param
        wd_params.append(self.conv_out_weight)
        return wd_params, nowd_params