import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def transposed_conv3d_kernel(
    x_ptr, w_ptr, out_ptr,
    stride_x, stride_w, stride_out,
    N, O_C, I_C, K_d, K_w, K_h,
    D, W, H, D_out, W_out, H_out,
    padding_d, padding_w, padding_h,
    stride_d, stride_w, stride_h,
    BLOCK_D: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_NOC: tl.constexpr
):
    # 1. Program coordinates
    pid_d = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_noc = tl.program_id(2)

    # 2. Output offsets for D and W
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # 3. Output offsets for N, O_C
    offs_noc = pid_noc * BLOCK_NOC + tl.arange(0, BLOCK_NOC)

    # Map 1D N_O_C index to 2D N, O_C
    N_OC = N * O_C
    o_c = offs_noc % O_C
    n = offs_noc // O_C

    # 4. Height offsets (H_out is fully covered in registers)
    offs_h = tl.arange(0, H_out)

    # 5. Initialize accumulator
    accum = tl.zeros((BLOCK_D, BLOCK_W, BLOCK_NOC, H_out), dtype=tl.float32)

    # 6. Loop over Input Channels (I_C), Kernel Depth (K_d), and Kernel Width (K_w)
    for ic in range(I_C):
        for kd in range(K_d):
            for kw in range(K_w):
                # Calculate corresponding input spatial indices
                # d_in = d_out * stride_d + kd - padding_d
                d_in = offs_d * stride_d + kd - padding_d
                w_in = offs_w * stride_w + kw - padding_w

                # h_in = h_out * stride_h + kh - padding_h
                # Since we fully tile H_out, we can compute h_in for all H_out simultaneously
                # We will handle K_h by iterating or masking. Let's iterate K_h inside to keep tiling clean.

                for kh in range(K_h):
                    h_in = offs_h * stride_h + kh - padding_h

                    # Masks for valid input coordinates
                    mask_d = (d_in >= 0) & (d_in < D)
                    mask_w = (w_in >= 0) & (w_in < W)
                    mask_h = (h_in >= 0) & (h_in < H)
                    mask_noc = offs_noc < N_OC

                    # Combine masks
                    mask = mask_d[:, None, None, None] & \
                           mask_w[None, :, None, None] & \
                           mask_noc[None, None, :, None] & \
                           mask_h[None, None, None, :]

                    # Load Input (x)
                    # x layout: [N, I_C, D, W, H]
                    # We need: x[n, ic, d_in, w_in, h_in]
                    x_idx = (n[:, None] * I_C * D * W * H) + \
                            (ic * D * W * H) + \
                            (d_in[None, :, None, None] * W * H) + \
                            (w_in[:, None, None, None] * H) + \
                            (h_in[None, None, None, :])

                    x = tl.load(x_ptr + x_idx, mask=mask, other=0.0)

                    # Load Weight (w)
                    # w layout: [O_C, I_C, K_d, K_w, K_h]
                    # We need: w[o_c, ic, kd, kw, kh]
                    w_idx = (o_c * I_C * K_d * K_w * K_h) + \
                            (ic * K_d * K_w * K_h) + \
                            (kd * K_w * K_h) + \
                            (kw * K_h) + \
                            (kh)

                    w = tl.load(w_ptr + w_idx, mask=mask_noc[:, None], other=0.0)

                    # Accumulate: x * w
                    # x: (BLOCK_D, BLOCK_W, BLOCK_NOC, H_out)
                    # w: (BLOCK_NOC, 1) -> broadcast
                    accum += x * w[:, None, :, None]

    # 7. Store Output
    # out layout: [N, O_C, D_out, W_out, H_out]
    out_idx = (n[:, None] * O_C * D_out * W_out * H_out) + \
              (o_c[None, :, None, None] * D_out * W_out * H_out) + \
              (offs_d[:, None, None, None] * W_out * H_out) + \
              (offs_w[None, :, None, None] * H_out) + \
              (offs_h[None, None, None, :])

    out_mask = (offs_d[:, None, None, None] < D_out) & \
               (offs_w[None, :, None, None] < W_out) & \
               (offs_noc[None, None, :, None] < N_OC) & \
               (offs_h[None, None, None, :] < H_out)

    tl.store(out_ptr + out_idx, accum, mask=out_mask)


def transposed_conv3d_torch(x: torch.Tensor, w: torch.Tensor, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0)):
    """
    Custom Triton implementation of 3D Transposed Convolution.
    Assumes groups=1, bias=False.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    N, I_C, D, W, H = x.shape
    O_C, _, K_d, K_w, K_h = w.shape

    stride_d, stride_w, stride_h = stride
    padding_d, padding_w, padding_h = padding
    output_padding_d, output_padding_w, output_padding_h = output_padding

    # Calculate output dimensions
    D_out = (D - 1) * stride_d - 2 * padding_d + K_d + output_padding_d
    W_out = (W - 1) * stride_w - 2 * padding_w + K_w + output_padding_w
    H_out = (H - 1) * stride_h - 2 * padding_h + K_h + output_padding_h

    out = torch.empty((N, O_C, D_out, W_out, H_out), dtype=torch.float32, device=x.device)

    # Block sizes
    BLOCK_D = 32
    BLOCK_W = 32
    BLOCK_NOC = 16

    grid = (
        triton.cdiv(D_out, BLOCK_D),
        triton.cdiv(W_out, BLOCK_W),
        triton.cdiv(N * O_C, BLOCK_NOC)
    )

    transposed_conv3d_kernel[grid](
        x, w, out,
        x.stride(), w.stride(), out.stride(),
        N, O_C, I_C, K_d, K_w, K_h,
        D, W, H, D_out, W_out, H_out,
        padding_d, padding_w, padding_h,
        stride_d, stride_w, stride_h,
        BLOCK_D, BLOCK_W, BLOCK_NOC
    )

    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert groups == 1 and bias == False, "Custom Triton kernel currently optimized for groups=1 and bias=False"

        # Initialize weights in [O_C, I_C, K_d, K_w, K_h] format
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2], dtype=torch.float32))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton.
        """
        return transposed_conv3d_torch(x, self.weight, stride=self.stride, padding=self.padding, output_padding=self.output_padding)