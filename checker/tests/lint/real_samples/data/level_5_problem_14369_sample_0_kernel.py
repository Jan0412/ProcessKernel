import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel that extracts 2x2 patches and concatenates the channel dim
# ----------------------------------------------------------------------
@triton.jit
def patch_merge_kernel(
    x_ptr,          # input pointer (B, D, H, W, C)
    out_ptr,        # output pointer (B, D, H//2, W//2, 4*C)
    B, D, H, W, C,  # input dimensions
    out_H, out_W,   # output spatial dimensions
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,   # how many channels we load per iteration (compile‑time)
):
    # ------------------------------------------------------------------
    # 1) Compute a linear index for each (b, d, ho, wo) position this program handles
    # ------------------------------------------------------------------
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (B * D * out_H * out_W)

    # Decode linear index into (b, d, ho, wo)
    n_pos = out_H * out_W
    bd = offs // n_pos                # combined batch*depth index
    pos = offs % n_pos
    ho = pos // out_W
    wo = pos % out_W

    b = bd // D
    d = bd % D

    # ------------------------------------------------------------------
    # 2) Compute base offsets for the four source locations
    # ------------------------------------------------------------------
    # Strides for the input tensor (contiguous, last dim = C)
    stride_B = D * H * W * C
    stride_D = H * W * C
    stride_H = W * C
    stride_W = C
    # base offset for each (b,d) slice
    base = b * stride_B + d * stride_D

    # Offsets for the four spatial positions (2*ho +/- 1, 2*wo +/- 1)
    x0_off = base + (2 * ho) * stride_H + (2 * wo) * stride_W
    x1_off = base + (2 * ho + 1) * stride_H + (2 * wo) * stride_W
    x2_off = base + (2 * ho) * stride_H + (2 * wo + 1) * stride_W
    x3_off = base + (2 * ho + 1) * stride_H + (2 * wo + 1) * stride_W

    # ------------------------------------------------------------------
    # 3) Load a block of channels from each of the four positions
    # ------------------------------------------------------------------
    c_off = tl.arange(0, BLOCK_C)
    c_mask = c_off < C

    x0 = tl.load(x_ptr + x0_off + c_off, mask=c_mask, other=0.0)
    x1 = tl.load(x_ptr + x1_off + c_off, mask=c_mask, other=0.0)
    x2 = tl.load(x_ptr + x2_off + c_off, mask=c_mask, other=0.0)
    x3 = tl.load(x_ptr + x3_off + c_off, mask=c_mask, other=0.0)

    # ------------------------------------------------------------------
    # 4) Store them contiguously as [x0, x1, x2, x3] in the output tensor
    # ------------------------------------------------------------------
    # Output strides (contiguous, last dim = 4*C)
    out_stride_B = D * out_H * out_W * (4 * C)
    out_stride_D = out_H * out_W * (4 * C)
    out_stride_H = out_W * (4 * C)
    out_stride_W = 4 * C
    out_base = b * out_stride_B + d * out_stride_D + ho * out_stride_H + wo * out_stride_W

    # Write the four blocks
    tl.store(out_ptr + out_base + c_off,                     x0, mask=c_mask)          # channel block 0..C-1
    tl.store(out_ptr + out_base + C + c_off,                 x1, mask=c_mask)          # channel block C..2C-1
    tl.store(out_ptr + out_base + 2 * C + c_off,             x2, mask=c_mask)          # channel block 2C..3C-1
    tl.store(out_ptr + out_base + 3 * C + c_off,             x3, mask=c_mask)          # channel block 3C..4C-1


def triton_patch_merge(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that launches the Triton kernel to perform the PatchMerging step
    (spatial down‑sampling by 2 and channel concatenation).
    Input shape : (B, D, H, W, C)  – must be contiguous on GPU.
    Output shape: (B, D, H//2, W//2, 4*C)
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    x = x.contiguous()
    B, D, H, W, C = x.shape
    out_H = H // 2
    out_W = W // 2

    out = torch.empty((B, D, out_H, out_W, 4 * C), dtype=x.dtype, device=x.device)

    total = B * D * out_H * out_W
    BLOCK_SIZE = 128            # programs per launch
    BLOCK_C = 32                # must be >= C (C=4 in our use‑case)

    grid = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    patch_merge_kernel[grid](
        x,
        out,
        B, D, H, W, C,
        out_H, out_W,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_C=BLOCK_C,
    )
    return out


# ----------------------------------------------------------------------
# Optimized PatchMerging module (ModelNew)
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor of shape (B, D, H, W, C) where C == self.dim
        """
        B, D, H, W, C = x.shape
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # Pad the height and width dimensions to make them even
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            H, W = x.shape[2], x.shape[3]

        # Triton‑accelerated patch merging (down‑sample + channel concat)
        x = triton_patch_merge(x)          # (B, D, H//2, W//2, 4*C)

        # Remaining operations stay as PyTorch for simplicity
        x = self.norm(x)
        x = self.reduction(x)
        return x