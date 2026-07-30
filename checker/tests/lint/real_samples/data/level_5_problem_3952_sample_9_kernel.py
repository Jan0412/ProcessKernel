import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------

@triton.jit
def maxpool2d_kernel(
    x_ptr,          # input pointer (N*C*H*W)
    out_ptr,        # output pointer (N*C*H_out*W_out)
    N, C, H, W,     # input dimensions
    K,              # kernel (and stride) size
    H_out, W_out,   # output spatial dimensions
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < (N * C * H_out * W_out)

    # decode linear offset into (n, c, ho, wo)
    tmp = offs
    wo = tmp % W_out
    tmp = tmp // W_out
    ho = tmp % H_out
    tmp = tmp // H_out
    c = tmp % C
    n = tmp // C

    # base offset for the top‑left corner of the pooling window
    base = ((n * C + c) * H + ho * K) * W + wo * K

    # compute max over KxK window
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    for kh in range(K):
        for kw in range(K):
            idx = base + kh * W + kw
            val = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offs, max_val, mask=mask)


def triton_maxpool2d(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """
    MaxPool2d with square kernel and stride == kernel (no padding).
    """
    assert x.is_cuda, "input must be on CUDA"
    x = x.contiguous()
    N, C, H, W = x.shape
    assert H % kernel == 0 and W % kernel == 0, "spatial dims must be divisible by kernel"
    H_out, W_out = H // kernel, W // kernel

    out = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)

    n_elements = N * C * H_out * W_out
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    maxpool2d_kernel[grid](
        x, out,
        N, C, H, W,
        kernel,
        H_out, W_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def add3_kernel(
    a_ptr, b_ptr, c_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    c = tl.load(c_ptr + offs, mask=mask, other=0.0)

    out = a + b + c
    tl.store(out_ptr + offs, out, mask=mask)


def triton_add3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Element‑wise addition of three tensors."""
    assert a.is_cuda and b.is_cuda and c.is_cuda, "All tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    out = torch.empty_like(a)

    n_elements = a.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    add3_kernel[grid](
        a, b, c,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # keep the original layers – only the pooling and final addition are replaced
        self.pool1 = nn.MaxPool2d(kernel_size=1)   # kept for API compatibility (not used)
        self.pool2 = nn.MaxPool2d(kernel_size=2)   # kept for API compatibility (not used)
        self.pool3 = nn.MaxPool2d(kernel_size=4)   # kept for API compatibility (not used)

        self.deconv2 = nn.ConvTranspose2d(
            512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
        )
        self.deconv3 = nn.ConvTranspose2d(
            512, 512, kernel_size=(6, 6), stride=(4, 4), padding=(1, 1)
        )

    def forward(self, x):
        # ---- replace three MaxPool2d with Triton kernels ----
        x1 = triton_maxpool2d(x, 1)   # identity
        x2 = triton_maxpool2d(x, 2)
        x3 = triton_maxpool2d(x, 4)

        # ---- keep the transposed convolutions as they are ----
        x2 = self.deconv2(x2)
        x3 = self.deconv3(x3)

        # ---- fused addition of three branches ----
        out = triton_add3(x1, x2, x3)
        return out


# ------------------------------------------------------------
# Export the model name expected by the benchmark harness
# ------------------------------------------------------------
Model = ModelNew