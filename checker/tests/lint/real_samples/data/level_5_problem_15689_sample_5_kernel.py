import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------- Triton kernel for concatenation --------------------
@triton.jit
def cat_kernel(
    out_ptr,  # output base pointer
    x1_ptr, x2_ptr, x3_ptr, x4_ptr,  # input pointers
    c1, c2, c3, c4,                # channel sizes of each input
    H, W,                          # spatial dimensions
    N, total_c,                    # batch size and total channels
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * total_c * H * W

    # Compute indices in NCHW layout
    hw = H * W
    nc = total_c * hw
    n = offsets // nc
    rem = offsets % nc
    c = rem // hw
    hw_idx = rem % hw
    h = hw_idx // W
    w = hw_idx % W

    # Offsets into each source tensor
    src_offset = tl.where(
        c < c1,
        (n * c1 + c) * hw + h * W + w,
        tl.where(
            c < c1 + c2,
            (n * c2 + (c - c1)) * hw + h * W + w,
            tl.where(
                c < c1 + c2 + c3,
                (n * c3 + (c - c1 - c2)) * hw + h * W + w,
                (n * c4 + (c - c1 - c2 - c3)) * hw + h * W + w,
            ),
        ),
    )

    src_ptr = tl.where(
        c < c1,
        x1_ptr,
        tl.where(
            c < c1 + c2,
            x2_ptr,
            tl.where(c < c1 + c2 + c3, x3_ptr, x4_ptr),
        ),
    )
    val = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def triton_cat(tensors):
    """
    Concatenates a list of 4 tensors along channel dimension using a Triton kernel.
    All tensors must have identical batch, height and width dimensions.
    """
    assert len(tensors) == 4, "triton_cat expects exactly 4 tensors."
    for t in tensors:
        assert t.is_cuda and t.is_contiguous(), "All tensors must be CUDA and contiguous."

    x1, x2, x3, x4 = tensors
    N, C1, H, W = x1.shape
    _, C2, _, _ = x2.shape
    _, C3, _, _ = x3.shape
    _, C4, _, _ = x4.shape

    total_c = C1 + C2 + C3 + C4
    out = torch.empty((N, total_c, H, W), dtype=x1.dtype, device=x1.device)

    n_elements = N * total_c * H * W
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    cat_kernel[grid](
        out,
        x1,
        x2,
        x3,
        x4,
        C1,
        C2,
        C3,
        C4,
        H,
        W,
        N,
        total_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# -------------------- Original convolution helper --------------------
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
         dilation=1, groups=1):
    """standard convolution with padding (bias disabled)"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


# -------------------- Optimized model --------------------
class PyConv4(nn.Module):
    def __init__(
        self,
        inplans,
        planes,
        pyconv_kernels=[3, 5, 7, 9],
        stride=1,
        pyconv_groups=[1, 4, 8, 16],
    ):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(
            inplans,
            planes // 4,
            kernel_size=pyconv_kernels[0],
            padding=pyconv_kernels[0] // 2,
            stride=stride,
            groups=pyconv_groups[0],
        )
        self.conv2_2 = conv(
            inplans,
            planes // 4,
            kernel_size=pyconv_kernels[1],
            padding=pyconv_kernels[1] // 2,
            stride=stride,
            groups=pyconv_groups[1],
        )
        self.conv2_3 = conv(
            inplans,
            planes // 4,
            kernel_size=pyconv_kernels[2],
            padding=pyconv_kernels[2] // 2,
            stride=stride,
            groups=pyconv_groups[2],
        )
        self.conv2_4 = conv(
            inplans,
            planes // 4,
            kernel_size=pyconv_kernels[3],
            padding=pyconv_kernels[3] // 2,
            stride=stride,
            groups=pyconv_groups[3],
        )

    def forward(self, x):
        # Compute each branch
        y1 = self.conv2_1(x)
        y2 = self.conv2_2(x)
        y3 = self.conv2_3(x)
        y4 = self.conv2_4(x)
        # Concatenate using Triton kernel
        return triton_cat([y1, y2, y3, y4])


# Export the optimized architecture under the required name
ModelNew = PyConv4