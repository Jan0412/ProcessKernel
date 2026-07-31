import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# ==============================================================================
# Depthwise Convolution Kernel
# ==============================================================================

@triton.jit
def depthwise_conv2d_kernel(
    X, W, B, Y,
    N, C, H, W,
    KH, KW,
    SH, SW,
    PH, PW,
    DH, DW,
    OH, OW,
    stride_x, stride_w, stride_y,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    # Iterate over channels in blocks
    for c_off in range(0, C, BLOCK_C):
        # Iterate over output spatial dimensions in blocks
        for oh_off in range(0, OH, BLOCK_OH):
            for ow_off in range(0, OW, BLOCK_OW):

                # Current output coordinates
                oh = pid_oh * BLOCK_OH + oh_off
                ow = pid_ow * BLOCK_OW + ow_off

                # Check bounds
                if oh >= OH or ow >= OW:
                    continue

                # Input coordinates for the top-left of the kernel window
                ih = oh * SH - PH
                iw = ow * SW - PW

                # Load weights for the current channel group
                # Weights shape: (C, 1, KH, KW) -> effectively (C, KH, KW)
                # We need to load W[c, kh, kw] for the current c

                # We will accumulate the result for the current output position
                acc = tl.zeros([BLOCK_N, BLOCK_C], dtype=tl.float32)

                # Load Bias if present
                # Bias shape: (C,)
                # We only add it if it's the first block of channels for this output position
                # To simplify, we'll add bias at the end or handle it carefully.
                # Given the complexity, let's assume bias is added after the sum or handled per-channel.
                # For this kernel, we'll sum up the convolution.

                for kh in range(KH):
                    for kw in range(KW):

                        # Calculate input spatial coordinates
                        h_in = ih + kh * DH
                        w_in = iw + kw * DW

                        # Masks for input bounds
                        mask_h = (h_in >= 0) & (h_in < H)
                        mask_w = (w_in >= 0) & (w_in < W)

                        # Load Input: X[N, C, H, W]
                        # We need to load X[n, c, h_in, w_in]
                        # Create offsets for N and C
                        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                        c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

                        # Flatten indices for loading
                        # X is NCHW. 
                        # x_ptr = X + n*stride_n + c*stride_c + h*stride_h + w*stride_w

                        # Create a mask for N and C bounds
                        mask_n = n_offsets < N
                        mask_c = c_offsets < C

                        # Combined mask
                        mask = mask_n[:, None] & mask_c[None, :] & mask_h & mask_w

                        # Input pointers
                        x_ptrs = X + (n_offsets[:, None] * stride_x[0]) + \
                                     (c_offsets[None, :] * stride_x[1]) + \
                                     (h_in * stride_x[2]) + \
                                     (w_in * stride_x[3])

                        x_val = tl.load(x_ptrs, mask=mask, other=0.0)

                        # Load Weight: W[C, KH, KW]
                        # w_ptr = W + c*stride_c + kh*stride_kh + kw*stride_kw
                        w_ptrs = W + (c_offsets * stride_w[0]) + \
                                     (kh * stride_w[1]) + \
                                     (kw * stride_w[2])

                        # Weight mask (only C dimension matters here)
                        w_mask = c_offsets < C

                        w_val = tl.load(w_ptrs, mask=w_mask, other=0.0)[None, :]

                        # Accumulate
                        acc += x_val * w_val

                # Add Bias
                if B is not None:
                    b_ptrs = B + c_offsets
                    b_mask = c_offsets < C
                    b_val = tl.load(b_ptrs, mask=b_mask, other=0.0)[None, :]
                    acc += b_val

                # Store Output: Y[N, C, OH, OW]
                y_ptrs = Y + (n_offsets[:, None] * stride_y[0]) + \
                             (c_offsets[None, :] * stride_y[1]) + \
                             (oh * stride_y[2]) + \
                             (ow * stride_y[3])

                y_mask = mask_n[:, None] & mask_c[None, :]

                tl.store(y_ptrs, acc, mask=y_mask)

# ==============================================================================
# Pointwise Convolution Kernel (GEMM)
# ==============================================================================

@triton.jit
def pointwise_conv2d_kernel(
    X, W, B, Y,
    N, C, H, W,
    OC,
    stride_x, stride_w, stride_y,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Each program handles a tile of the output: (N, OC)
    # The spatial dimensions (H, W) are treated as part of the batch dimension in the GEMM logic
    # effectively mapping (N, H, W) -> M, and C -> K, OC -> N

    # Map program IDs to output coordinates
    pid_m = tl.program_id(0) # Corresponds to N * H * W
    pid_n = tl.program_id(1) # Corresponds to OC

    # Calculate M, K, N dimensions for the GEMM
    # M = N * H * W
    # K = C
    # N = OC

    # Offsets for the current block
    m_offsets = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)
    n_offsets = pid_n * BLOCK_OC + tl.arange(0, BLOCK_OC)

    # Masks
    m_mask = m_offsets < (N * H * W)
    n_mask = n_offsets < OC

    # We need to iterate over K (C)
    acc = tl.zeros([BLOCK_N, BLOCK_OC], dtype=tl.float32)

    for c_off in range(0, C, BLOCK_C):
        c_offsets = c_off + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C

        # Load X: Shape (N, C, H, W) -> Viewed as (M, K)
        # We need to map m_offsets back to (n_idx, h_idx, w_idx)
        # m_idx = n_idx * H * W + h_idx * W + w_idx

        n_idx = m_offsets // (H * W)
        h_idx = (m_offsets % (H * W)) // W
        w_idx = m_offsets % W

        # X pointers
        x_ptrs = X + (n_idx[:, None] * stride_x[0]) + \
                     (c_offsets[None, :] * stride_x[1]) + \
                     (h_idx[:, None] * stride_x[2]) + \
                     (w_idx[:, None] * stride_x[3])

        x_mask = m_mask[:, None] & c_mask[None, :]

        x_val = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W: Shape (OC, C) -> Viewed as (N, K)
        # W[oc, c]
        w_ptrs = W + (n_offsets[:, None] * stride_w[0]) + \
                     (c_offsets[None, :] * stride_w[1])

        w_mask = n_mask[:, None] & c_mask[None, :]

        w_val = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x_val, w_val.T)

    # Add Bias
    if B is not None:
        b_ptrs = B + n_offsets
        b_mask = n_mask
        b_val = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += b_val[None, :]

    # Store Y: Shape (N, OC, H, W)
    # Map m_offsets back to (n_idx, h_idx, w_idx) for output indexing
    n_idx = m_offsets // (H * W)
    h_idx = (m_offsets % (H * W)) // W
    w_idx = m_offsets % W

    y_ptrs = Y + (n_idx[:, None] * stride_y[0]) + \
                 (n_offsets[None, :] * stride_y[1]) + \
                 (h_idx[:, None] * stride_y[2]) + \
                 (w_idx[:, None] * stride_y[3])

    y_mask = m_mask[:, None] & n_mask[None, :]

    tl.store(y_ptrs, acc, mask=y_mask)


# ==============================================================================
# Wrapper Functions
# ==============================================================================

def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    # x: (N, C, H, W)
    # weight: (C, 1, KH, KW)
    # bias: (C,)

    N, C, H, W = x.shape
    KH, KW = weight.shape[2], weight.shape[3]

    OH = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1

    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    # Grid dimensions
    # We process N, C, OH, OW. To keep block sizes reasonable, we might need to chunk.
    # For simplicity in this implementation, we use a grid that covers the full dimensions
    # but relies on the kernel's internal loops to handle blocks if they are large.
    # However, Triton grid should be small enough. Let's use fixed block sizes.

    BLOCK_N = 1
    BLOCK_C = 32
    BLOCK_OH = 16
    BLOCK_OW = 16

    grid = (
        (N + BLOCK_N - 1) // BLOCK_N,
        (C + BLOCK_C - 1) // BLOCK_C,
        (OH + BLOCK_OH - 1) // BLOCK_OH,
        (OW + BLOCK_OW - 1) // BLOCK_OW
    )

    # Strides
    stride_x = (x.stride(0), x.stride(1), x.stride(2), x.stride(3))
    stride_w = (weight.stride(0), weight.stride(2), weight.stride(3))
    stride_y = (out.stride(0), out.stride(1), out.stride(2), out.stride(3))

    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        N, C, H, W,
        KH, KW,
        stride, stride,
        padding, padding,
        dilation, dilation,
        OH, OW,
        stride_x, stride_w, stride_y,
        BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C, BLOCK_OH=BLOCK_OH, BLOCK_OW=BLOCK_OW
    )

    return out

def triton_pointwise_conv2d(x, weight, bias=None):
    # x: (N, C, H, W)
    # weight: (OC, C)
    # bias: (OC,)

    N, C, H, W = x.shape
    OC, _ = weight.shape

    out = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)

    BLOCK_N = 64
    BLOCK_OC = 64
    BLOCK_C = 32

    # Grid covers M (N*H*W) and N (OC)
    M = N * H * W
    grid = (
        (M + BLOCK_N - 1) // BLOCK_N,
        (OC + BLOCK_OC - 1) // BLOCK_OC
    )

    stride_x = (x.stride(0), x.stride(1), x.stride(2), x.stride(3))
    stride_w = (weight.stride(0), weight.stride(1))
    stride_y = (out.stride(0), out.stride(1), out.stride(2), out.stride(3))

    pointwise_conv2d_kernel[grid](
        x, weight, bias, out,
        N, C, H, W,
        OC,
        stride_x, stride_w, stride_y,
        BLOCK_N=BLOCK_N, BLOCK_OC=BLOCK_OC, BLOCK_C=BLOCK_C
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # We need to register the weights and biases as parameters to match the original model's state dict
        # Depthwise weights: (in_channels, 1, kernel_size, kernel_size)
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        # Pointwise weights: (out_channels, in_channels)
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels))

        if bias:
            self.depthwise_bias = nn.Parameter(torch.zeros(in_channels))
            self.pointwise_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('depthwise_bias', None)
            self.register_parameter('pointwise_bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise Convolution
        x = triton_depthwise_conv2d(
            x, 
            self.depthwise_weight, 
            self.depthwise_bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )

        # Pointwise Convolution
        x = triton_pointwise_conv2d(
            x, 
            self.pointwise_weight, 
            self.pointwise_bias
        )

        return x