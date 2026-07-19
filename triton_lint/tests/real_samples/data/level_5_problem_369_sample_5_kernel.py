import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------- Triton kernel for horizontal flip ------------------- #
@triton.jit
def hflip_kernel(
    inp_ptr,          # pointer to input tensor
    out_ptr,          # pointer to output tensor
    stride_row,       # number of elements in a row (i.e., width)
    width,            # width of the image (last dimension)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes ONE row (i.e. one (C,H) slice) of the tensor.
    Inside a row we copy BLOCK_SIZE elements at a time, reading from the
    reversed column index and writing to the forward column index.
    """
    row_idx = tl.program_id(0)                     # which row we are processing
    col = tl.arange(0, BLOCK_SIZE)                 # column offsets inside the row
    mask = col < width                              # guard for tail

    # reversed column index
    rev_col = width - 1 - col

    # compute absolute pointers
    inp = inp_ptr + row_idx * stride_row + rev_col
    out = out_ptr + row_idx * stride_row + col

    val = tl.load(inp, mask=mask, other=0.0)
    tl.store(out, val, mask=mask)


def triton_hflip(x: torch.Tensor) -> torch.Tensor:
    """
    Horizontal flip using a custom Triton kernel.
    Works for any tensor shape (..., C, H, W) as long as it is contiguous
    and resides on CUDA.
    """
    assert x.is_cuda, "triton_hflip only works on CUDA tensors"
    x = x.contiguous()
    out = torch.empty_like(x)

    width = x.shape[-1]                              # W
    rows = x.numel() // width                         # total number of rows = prod of all dims except W
    BLOCK_SIZE = 128                                 # tunable

    grid = (rows,)                                    # one program per row
    hflip_kernel[grid](
        x,
        out,
        width,          # stride_row = width for a contiguous tensor
        width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ------------------- Optimized model ------------------- #
class ModelNew(nn.Module):
    """
    Same interface as the original Hflip model but uses a Triton kernel
    for the horizontal flip operation.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return triton_hflip(input)

    def __repr__(self):
        return self.__class__.__name__