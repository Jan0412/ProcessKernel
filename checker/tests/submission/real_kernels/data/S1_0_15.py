import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    X, W, O,
    N, C, H, W,
    OC, KH, KW,
    stride, pad,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Grid dimensions: pid_n (Batch), pid_oc (OutChannel), pid_tile (Spatial Tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_tile = tl.program_id(2)

    # Calculate base output tile coordinates
    # We treat the output spatial dimensions (OH, OW) as a 1D array of tiles
    # OH = (H + 2*pad - KH) // stride + 1
    # OW = (W + 2*pad - KW) // stride + 1
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (W + 2 * pad - KW) // stride + 1

    num_tiles_h = (OH + BLOCK_H - 1) // BLOCK_H
    num_tiles_w = (OW + BLOCK_W - 1) // BLOCK_W
    total_tiles = num_tiles_h * num_tiles_w

    tile_idx = pid_tile
    tile_h_idx = tile_idx // num_tiles_w
    tile_w_idx = tile_idx % num_tiles_w

    # Base offsets for the tile
    oh_base = tile_h_idx * BLOCK_H
    ow_base = tile_w_idx * BLOCK_W

    # Offsets within the tile
    off_h = tl.arange(0, BLOCK_H)
    off_w = tl.arange(0, BLOCK_W)

    # Global output offsets
    out_h = oh_base + off_h[:, None]
    out_w = ow_base + off_w[None, :]

    # Masks for output spatial validity
    mask_h = out_h < OH
    mask_w = out_w < OW
    mask = mask_h & mask_w

    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over Input Channels and Kernel spatial dimensions
    for c in range(C):
        for kh in range(KH):
            for kw in range(KW):
                # Calculate input spatial coordinates
                # x_h = out_h * stride + kh - pad
                in_h = out_h * stride + kh - pad
                in_w = out_w * stride + kw - pad

                # Input masks
                mask_in_h = (in_h >= 0) & (in_h < H)
                mask_in_w = (in_w >= 0) & (in_w < W)
                mask_in = mask_in_h & mask_in_w

                # Load weights: W[oc, c, kh, kw]
                w_val = tl.load(W + pid_oc * C * KH * KW + c * KH * KW + kh * KW + kw)

                # Load input: X[n, c, in_h, in_w]
                # Stride for X: N * C * H * W
                x_ptr = X + pid_n * C * H * W + c * H * W + in_h * W + in_w
                x_val = tl.load(x_ptr, mask=mask_in, other=0.0)

                # Accumulate
                acc += x_val * w_val

    # Output pointer: O[n, oc, out_h, out_w]
    # Stride for O: N * OC * OH * OW
    out_ptr = O + pid_n * OC * OH * OW + pid_oc * OH * OW + out_h * OW + out_w

    # Store result
    tl.store(out_ptr, acc, mask=mask)


def triton_conv2d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Custom Triton Conv2d for NCHW layout.
    Assumes square kernel, same padding on all sides, and dilation=1.
    """
    assert x.is_cuda and w.is_cuda
    assert x.dim() == 4
    assert w.dim() == 4
    assert dilation == 1
    assert groups == 1

    N, C, H, W = x.shape
    OC, IC, KH, KW = w.shape

    assert C == IC

    # Calculate output dimensions
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1

    # Initialize output
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    # Add bias if present
    if b is not None:
        out += b.view(1, OC, 1, 1)

    BLOCK_H = 16
    BLOCK_W = 16

    # Grid dimensions
    num_tiles_h = (OH + BLOCK_H - 1) // BLOCK_H
    num_tiles_w = (OW + BLOCK_W - 1) // BLOCK_W
    num_tiles = num_tiles_h * num_tiles_w

    grid = (N, OC, num_tiles)

    conv2d_kernel[grid](
        x, w, out,
        N, C, H, W,
        OC, KH, KW,
        stride, padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )

    return out


class Conv2DFunctional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2DFunctional, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return triton_conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding
        )


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Replace nn.Conv2d with our custom functional module
        self.conv1 = Conv2DFunctional(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        return x

def get_inputs():
    batch_size = 256
    return [torch.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [1000]

# Ensure the model and inputs are on CUDA for testing if needed, 
# though the prompt asks for the code block only.