import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -------------------------------------------------------------
# Triton kernel: nearest-neighbor upsample (1‑D)
# -------------------------------------------------------------
@triton.jit
def upsample_nearest_kernel(
    x_ptr,          # input pointer
    out_ptr,        # output pointer
    N, C, L_out,    # output shape
    in_stride_n,    # stride for N dimension in input
    in_stride_c,    # stride for C dimension in input
    in_L,           # original length
    scale_factor,   # integer scale factor
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C * L_out)

    # Decode linear offset into (n, c, pos_out)
    n = offsets // (C * L_out)
    rem = offsets % (C * L_out)
    c = rem // L_out
    pos_out = rem % L_out

    # source position (nearest neighbour)
    src_pos = pos_out // scale_factor

    # compute input pointer offset
    x_offset = n * in_stride_n + c * in_stride_c + src_pos
    out_offset = n * (C * L_out) + c * L_out + pos_out

    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    tl.store(out_ptr + out_offset, x, mask=mask)


def triton_upsample_nearest(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    Nearest‑neighbor upsample along the last dimension using a Triton kernel.
    Only integer scale factors are supported (the original model uses 1.0, 2.0, …).
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.ndim == 3, "Expected 3‑D tensor (N, C, L)"
    scale_int = int(round(scale_factor))
    assert abs(scale_factor - scale_int) < 1e-6, "Scale factor must be integer"

    N, C, L = x.shape
    L_out = L * scale_int

    out = torch.empty((N, C, L_out), dtype=x.dtype, device=x.device)

    # strides of the input (contiguous layout)
    stride_n, stride_c, stride_l = x.stride()
    # flatten strides for the kernel
    in_stride_n = stride_n
    in_stride_c = stride_c

    total = N * C * L_out
    BLOCK_SIZE = 1024

    grid = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    upsample_nearest_kernel[grid](
        x,
        out,
        N,
        C,
        L_out,
        in_stride_n,
        in_stride_c,
        L,
        scale_int,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# -------------------------------------------------------------
# Triton kernel: 1‑D convolution (stride=1, padding can be set)
# -------------------------------------------------------------
@triton.jit
def conv1d_kernel(
    x_ptr,          # input pointer (N, C_in, L_in)
    w_ptr,          # weight pointer (C_out, C_in, K)
    b_ptr,          # bias pointer (C_out) or nullptr
    out_ptr,        # output pointer (N, C_out, L_out)
    N, C_in, L_in,
    C_out, K,
    padding,
    in_stride_n,    # stride for N dimension in input
    in_stride_c,    # stride for C dimension in input
    out_stride_n,   # stride for N dimension in output
    out_stride_c,   # stride for C dimension in output
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C_out * (L_in + 2 * padding - K + 1))

    # Decode linear offset into (n, c_out, pos_out)
    n = offsets // (C_out * (L_in + 2 * padding - K + 1))
    rem = offsets % (C_out * (L_in + 2 * padding - K + 1))
    c_out = rem // (L_in + 2 * padding - K + 1)
    pos_out = rem % (L_in + 2 * padding - K + 1)

    # accumulate convolution result
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # loop over kernel width
    for k in range(K):
        # input position (with padding)
        pos_in = pos_out + k - padding
        # mask for valid input positions
        valid = (pos_in >= 0) & (pos_in < L_in)

        # pointer to input element for each lane
        x_offset = n * in_stride_n + tl.arange(0, BLOCK_SIZE) * 0  # placeholder
        # compute actual offsets only for valid lanes
        x_offset = n * in_stride_n + (tl.arange(0, BLOCK_SIZE) // (C_out * (L_in + 2 * padding - K + 1))) * 0  # dummy to keep shape
        # Better: compute per‑lane offset using broadcasting
        x_offset = n * in_stride_n + (c_out * 0) + pos_in * 0  # dummy – will be replaced below

        # Real offset calculation:
        # input linear index = n*in_stride_n + ci*in_stride_c + pos_in
        # we will loop over ci inside the next loop
        for ci in range(C_in):
            # weight offset
            w_offset = c_out * (C_in * K) + ci * K + k
            w = tl.load(w_ptr + w_offset)

            # input offset
            in_offset = n * in_stride_n + ci * in_stride_c + pos_in
            x = tl.load(x_ptr + in_offset, mask=valid, other=0.0)

            acc += w * x

    # add bias if present
    if b_ptr is not None:
        b = tl.load(b_ptr + c_out)
        acc += b

    # write output
    out_offset = n * out_stride_n + c_out * out_stride_c + pos_out
    tl.store(out_ptr + out_offset, acc, mask=mask)


def triton_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    padding: int = 1,
) -> torch.Tensor:
    """
    1‑D convolution using a Triton kernel.
    Assumes stride=1 and dilation=1.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be CUDA"
    if bias is not None:
        assert bias.is_cuda, "Bias must be CUDA"

    N, C_in, L_in = x.shape
    C_out, _, K = weight.shape
    L_out = L_in + 2 * padding - K + 1

    out = torch.empty((N, C_out, L_out), dtype=x.dtype, device=x.device)

    # strides (contiguous layout)
    in_stride_n, in_stride_c, _ = x.stride()
    out_stride_n, out_stride_c, _ = out.stride()

    total = N * C_out * L_out
    BLOCK_SIZE = 1024
    grid = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # bias pointer handling – Triton expects a pointer; we pass 0 if None
    bias_ptr = bias if bias is not None else torch.tensor([], dtype=x.dtype, device=x.device)

    conv1d_kernel[grid](
        x,
        weight,
        bias_ptr,
        out,
        N,
        C_in,
        L_in,
        C_out,
        K,
        padding,
        in_stride_n,
        in_stride_c,
        out_stride_n,
        out_stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# -------------------------------------------------------------
# New model using the Triton kernels
# -------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # keep a regular Conv1d so we can reuse its weight & bias (parameter registration)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             stride=1, padding=1, bias=True)

    def forward(self, x):
        # 1) nearest‑neighbor upsample (only integer scale factors are supported)
        x_up = triton_upsample_nearest(x, self.scale_factor)

        # 2) convolution (stride=1, padding=1)
        out = triton_conv1d(x_up, self.conv.weight, self.conv.bias, padding=1)
        return out