import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -------------------------------------------------
# Triton kernels for the Squeeze‑Excitation block
# -------------------------------------------------

@triton.jit
def se_avg_pool_kernel(
    x_ptr,                # input tensor (N, C, H, W)
    avg_ptr,              # output tensor (N, C)   -> global average per channel
    N, C, H, W,
    BLOCK_NC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    idx = tl.program_id(0)                     # linear index over N*C
    mask = idx < BLOCK_NC

    # ------------------------------------------------------------------
    # Compute (n, c) coordinates
    # ------------------------------------------------------------------
    n = idx // C
    c = idx % C

    # stride to the first element of this (n,c) plane
    base = ((n * C + c) * H * W)

    # ------------------------------------------------------------------
    # Reduce over the spatial dimensions H*W
    # ------------------------------------------------------------------
    sum_val = tl.float32(0.0)
    for off in range(0, H * W, BLOCK_HW):
        cur_off = tl.arange(0, BLOCK_HW) + off
        cur_mask = (cur_off < H * W) & mask
        x = tl.load(x_ptr + base + cur_off, mask=cur_mask, other=0.0)
        sum_val += tl.sum(x)

    avg = sum_val / (H * W)

    # store result
    tl.store(avg_ptr + idx, avg, mask=mask)


@triton.jit
def se_fc1_kernel(
    avg_ptr,               # (N, C) input
    w1_ptr,                # (R, C) weight
    b1_ptr,                # (R,)   bias
    hidden_ptr,            # (N, R) output (after ReLU)
    N, C, R,
    BLOCK_NR: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    idx = tl.program_id(0)                     # linear index over N*R
    mask = idx < BLOCK_NR

    n = idx // R
    r = idx % R

    # accumulate dot product over C
    dot = tl.float32(0.0)
    for off in range(0, C, BLOCK_C):
        cur_off = tl.arange(0, BLOCK_C) + off
        cur_mask = (cur_off < C) & mask
        a = tl.load(avg_ptr + n * C + cur_off, mask=cur_mask, other=0.0)
        w = tl.load(w1_ptr + r * C + cur_off, mask=cur_mask, other=0.0)
        dot += tl.sum(a * w)

    # add bias and ReLU
    dot = dot + tl.load(b1_ptr + r)
    dot = tl.maximum(dot, tl.float32(0.0))   # ReLU

    tl.store(hidden_ptr + idx, dot, mask=mask)


@triton.jit
def se_fc2_kernel(
    hidden_ptr,            # (N, R) input
    w2_ptr,                # (C, R) weight
    b2_ptr,                # (C,)   bias
    scale_ptr,             # (N, C) output (after sigmoid)
    N, C, R,
    BLOCK_NC: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    idx = tl.program_id(0)                     # linear index over N*C
    mask = idx < BLOCK_NC

    n = idx // C
    c = idx % C

    # accumulate dot product over R
    dot = tl.float32(0.0)
    for off in range(0, R, BLOCK_R):
        cur_off = tl.arange(0, BLOCK_R) + off
        cur_mask = (cur_off < R) & mask
        a = tl.load(hidden_ptr + n * R + cur_off, mask=cur_mask, other=0.0)
        w = tl.load(w2_ptr + c * R + cur_off, mask=cur_mask, other=0.0)
        dot += tl.sum(a * w)

    dot = dot + tl.load(b2_ptr + c)

    # sigmoid
    scale = 1.0 / (1.0 + tl.exp(-dot))

    tl.store(scale_ptr + idx, scale, mask=mask)


@triton.jit
def se_apply_scale_kernel(
    x_ptr,                 # (N, C, H, W) input
    scale_ptr,             # (N, C) scaling factor
    out_ptr,               # (N, C, H, W) output
    N, C, H, W,
    BLOCK_NCHW: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    idx = tl.program_id(0)                     # linear index over N*C*H*W
    mask = idx < BLOCK_NCHW

    # derive coordinates
    n = idx // (C * H * W)
    rem = idx % (C * H * W)
    c = rem // (H * W)
    hw = rem % (H * W)

    # load scale for (n,c)
    scale = tl.load(scale_ptr + n * C + c, mask=mask, other=0.0)

    # load input element
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)

    # write scaled output
    tl.store(out_ptr + idx, x * scale, mask=mask)


# -------------------------------------------------
# Helper wrappers around the kernels
# -------------------------------------------------
def triton_se_forward(x: torch.Tensor,
                      conv1: nn.Conv2d,
                      conv2: nn.Conv2d) -> torch.Tensor:
    """
    x : (N, C, H, W)  FP32 CUDA tensor
    conv1, conv2 : 1x1 Conv2d layers used in Squeeze‑Excitation
    Returns the SE‑scaled tensor.
    """
    assert x.is_cuda and x.dtype == torch.float32
    N, C, H, W = x.shape
    R = conv1.out_channels                     # reduced dimension

    # ------------------------------------------------------------------
    # 1) Global average pooling (N*C)
    # ------------------------------------------------------------------
    avg = torch.empty((N, C), dtype=torch.float32, device=x.device)
    total_nc = N * C
    BLOCK_NC = 128
    BLOCK_HW = 64
    grid = lambda meta: ((total_nc + meta["BLOCK_NC"] - 1) // meta["BLOCK_NC"],)
    se_avg_pool_kernel[grid](
        x, avg,
        N, C, H, W,
        BLOCK_NC=BLOCK_NC, BLOCK_HW=BLOCK_HW
    )

    # ------------------------------------------------------------------
    # 2) First FC (ReLU) -> (N, R)
    # ------------------------------------------------------------------
    hidden = torch.empty((N, R), dtype=torch.float32, device=x.device)
    total_nr = N * R
    BLOCK_NR = 128
    BLOCK_C = 64
    grid = lambda meta: ((total_nr + meta["BLOCK_NR"] - 1) // meta["BLOCK_NR"],)
    # reshape weights to 2‑D contiguous tensors
    w1 = conv1.weight.view(R, C).contiguous()
    b1 = conv1.bias.contiguous()
    se_fc1_kernel[grid](
        avg, w1, b1, hidden,
        N, C, R,
        BLOCK_NR=BLOCK_NR, BLOCK_C=BLOCK_C
    )

    # ------------------------------------------------------------------
    # 3) Second FC (sigmoid) -> (N, C) scale
    # ------------------------------------------------------------------
    scale = torch.empty((N, C), dtype=torch.float32, device=x.device)
    total_nc = N * C
    BLOCK_NC = 128
    BLOCK_R = 64
    grid = lambda meta: ((total_nc + meta["BLOCK_NC"] - 1) // meta["BLOCK_NC"],)
    w2 = conv2.weight.view(C, R).contiguous()
    b2 = conv2.bias.contiguous()
    se_fc2_kernel[grid](
        hidden, w2, b2, scale,
        N, C, R,
        BLOCK_NC=BLOCK_NC, BLOCK_R=BLOCK_R
    )

    # ------------------------------------------------------------------
    # 4) Apply scaling factor to the original tensor
    # ------------------------------------------------------------------
    out = torch.empty_like(x)
    total_nchw = N * C * H * W
    BLOCK_NCHW = 256
    BLOCK_SPATIAL = 64
    grid = lambda meta: ((total_nchw + meta["BLOCK_NCHW"] - 1) // meta["BLOCK_NCHW"],)
    se_apply_scale_kernel[grid](
        x, scale, out,
        N, C, H, W,
        BLOCK_NCHW=BLOCK_NCHW, BLOCK_SPATIAL=BLOCK_SPATIAL
    )
    return out


# -------------------------------------------------
# Optimized model definition (ModelNew)
# -------------------------------------------------
class ModelNew(nn.Module):
    """
    Re‑implementation of the original SqueezeExcitation block where
    the expensive operations (global average pooling, two 1×1 convolutions,
    sigmoid and the final element‑wise multiplication) are fused into
    custom Triton kernels.
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        num_reduced_channels = make_divisible(
            max(in_channels, 8) // reduction, 8
        )
        # keep the original Conv2d modules so that their parameters are
        # registered with the optimizer and can be loaded from checkpoints.
        self.fc1 = nn.Conv2d(in_channels, num_reduced_channels,
                             kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(num_reduced_channels, in_channels,
                             kernel_size=1, bias=True)
        # The original implementation used ReLU and Sigmoid internally;
        # they are now embedded inside the Triton kernels.
        self.in_channels = in_channels

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # The Triton implementation expects the input to be contiguous.
        inp = inp.contiguous()
        return triton_se_forward(inp, self.fc1, self.fc2)


# -----------------------------------------------------------------
# Helper – make_divisible (unchanged from the original script)
# -----------------------------------------------------------------
def make_divisible(v, divisor=8, min_value=None):
    """
    The channel number of each layer should be divisible by 8.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v