import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    x_ptr, 
    out_ptr, 
    # Input shape
    N, C, D, H, W,
    # Output shape
    ND, NH, NW,
    # Kernel parameters
    KD, KH, KW,
    # Strides
    SD, SH, SW,
    # Padding
    PD, PH, PW,
    # Strides for memory layout
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_ond, stride_onc, stride_ond, stride_onh, stride_onw,
    # Loop bounds
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Map each block to a specific output element
    pid = tl.program_id(0)

    if pid >= n_elements:
        return

    # Decode output indices from pid
    # Output shape is (N, C, ND, NH, NW)
    nw_idx = pid % NW
    pid //= NW
    nh_idx = pid % NH
    pid //= NH
    nd_idx = pid % ND
    pid //= ND
    c_idx = pid % C
    pid //= C
    n_idx = pid

    # Calculate starting coordinates in the input tensor
    # Input coordinates: start = out_idx * stride - padding
    start_d = nd_idx * SD - PD
    start_h = nh_idx * SH - PH
    start_w = nw_idx * SW - PW

    # Initialize sum accumulator
    sum_val = tl.zeros([1], dtype=tl.float32)

    # Loop over the kernel window
    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                # Calculate input coordinates
                d = start_d + kd
                h = start_h + kh
                w = start_w + kw

                # Boundary check
                if d >= 0 and d < D and h >= 0 and h < H and w >= 0 and w < W:
                    # Calculate memory offset
                    offset = (n_idx * stride_xn + 
                              c_idx * stride_xc + 
                              d * stride_xd + 
                              h * stride_xh + 
                              w * stride_xw)
                    # Load and accumulate
                    val = tl.load(x_ptr + offset)
                    sum_val += val

    # Compute average
    avg = sum_val / (KD * KH * KW)

    # Calculate output memory offset and store
    out_offset = (n_idx * stride_ond + 
                  c_idx * stride_onc + 
                  nd_idx * stride_ond + 
                  nh_idx * stride_onh + 
                  nw_idx * stride_onw)

    tl.store(out_ptr + out_offset, avg)


def triton_avg_pool3d(x, kernel_size, stride, padding):
    """
    Wrapper for the Triton AvgPool3d kernel.
    """
    # Ensure contiguous memory for predictable strides
    x = x.contiguous()

    N, C, D, H, W = x.shape
    KD, KH, KW = kernel_size, kernel_size, kernel_size

    SD, SH, SW = stride, stride, stride
    PD, PH, PW = padding, padding, padding

    # Calculate output dimensions
    ND = (D + 2 * PD - KD) // SD + 1
    NH = (H + 2 * PH - KH) // SH + 1
    NW = (W + 2 * PW - KW) // SW + 1

    # Allocate output tensor
    out = torch.empty((N, C, ND, NH, NW), device=x.device, dtype=x.dtype)

    # Calculate total number of output elements
    n_elements = N * C * ND * NH * NW

    # Grid configuration: 1D grid, one block per output element
    # BLOCK_SIZE is not strictly used for tiling here as we do 1:1 mapping, 
    # but Triton requires it for the launch grid.
    BLOCK_SIZE = 1
    grid = (n_elements,)

    # Launch kernel
    avg_pool3d_kernel[grid](
        x, out,
        N, C, D, H, W,
        ND, NH, NW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton for 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)