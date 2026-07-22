import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_pad_kernel(
    inp_ptr,          # *float32 input pointer
    out_ptr,          # *float32 output pointer
    N, C, H, W,       # input dimensions
    H_out, W_out,     # output spatial dimensions
    stride: tl.constexpr,
    channel_offset: tl.constexpr,   # = C // 2
    BLOCK_SIZE: tl.constexpr,       # number of output elements per program
):
    # ---------- spatial index ----------
    pid = tl.program_id(0)          # flattened block index over N*H_out*W_out
    c = tl.program_id(1)            # central channel index (0 .. C-1)

    # linear indices for the block
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)

    total_spatial = N * H_out * W_out
    mask = offsets < total_spatial

    # decode (n, h_out, w_out) from linear offsets
    n = offsets // (H_out * W_out)
    tmp = offsets % (H_out * W_out)
    h_out = tmp // W_out
    w_out = tmp % W_out

    # ---------- compute average pool ----------
    # top‑left corner of the 3×3 window in the input (padding = 1)
    h_in = h_out * stride - 1
    w_in = w_out * stride - 1

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for dh in range(3):
        for dw in range(3):
            hi = h_in + dh
            wi = w_in + dw

            # bounds check for the input tensor
            valid_h = (hi >= 0) & (hi < H)
            valid_w = (wi >= 0) & (wi < W)
            valid = valid_h & valid_w

            # compute linear index into input
            inp_idx = ((n * C + c) * H + hi) * W + wi
            val = tl.load(inp_ptr + inp_idx, mask=valid & mask, other=0.0)
            sum_val += val

    avg = sum_val / 9.0

    # ---------- write to output (zero‑padded elsewhere) ----------
    out_c = channel_offset + c                     # central region channel
    out_idx = ((n * (2 * C) + out_c) * H_out + h_out) * W_out + w_out
    tl.store(out_ptr + out_idx, avg, mask=mask)


def triton_avg_pool_pad(x: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Performs the RevPaddingLayer operation (AvgPool2d + zero‑padding in channel dim)
    using a single fused Triton kernel.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    x = x.contiguous()

    N, C, H, W = x.shape
    assert C % 2 == 0, "Channel dimension must be even for RevPaddingLayer"

    # output spatial size (same formula as nn.AvgPool2d with kernel=3, padding=1)
    H_out = (H + 2 * 1 - 3) // stride + 1
    W_out = (W + 2 * 1 - 3) // stride + 1

    out = torch.zeros((N, 2 * C, H_out, W_out), dtype=x.dtype, device=x.device)

    total_spatial = N * H_out * W_out
    BLOCK_SIZE = 128

    grid = (
        ( (total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE, ),  # program_id(0)
        (C, ),                                              # program_id(1) – central channels
    )

    avg_pool_pad_kernel[grid](
        x,
        out,
        N, C, H, W,
        H_out, W_out,
        stride,
        C // 2,               # channel_offset
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of RevPaddingLayer using a fused Triton kernel.
    """
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool_pad(x, self.stride)


# ----------------------------------------------------------------------
# Helper functions (kept identical to the original benchmark harness)
def get_inputs():
    # Example input matching the original architecture
    return [torch.rand([4, 4, 4, 4], device="cuda")]


def get_init_inputs():
    # The original code expects a list with a single integer (stride)
    return [1]