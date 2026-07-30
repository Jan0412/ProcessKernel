import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------- Triton kernels -------------------- #

@triton.jit
def gather_kernel(
    feat_ptr,          # *float32, input feature map (B, C, H, W)
    ind_ptr,           # *int32,   indices (B, M)
    out_ptr,           # *float32, output (B, M, C)
    B, C, H, W, M,
    stride_f_bc,       # stride for batch dimension in feat
    stride_f_c,        # stride for channel dimension in feat
    stride_f_h,        # stride for height dimension in feat
    stride_f_w,        # stride for width dimension in feat
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)                      # one program per (b, m)
    b = pid // M
    m = pid % M

    # load the flat index and decompose to (h, w)
    ind = tl.load(ind_ptr + b * M + m)
    idx = ind.to(tl.int32)
    w = idx % W
    h = idx // W

    # channel block
    c_start = tl.program_id(1) * BLOCK_SIZE
    offs = c_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < C

    # address of input: ((b*C + (c_start+offs)) * H + h) * W + w
    inp_ptr = (
        feat_ptr
        + b * stride_f_bc
        + offs * stride_f_c
        + h * stride_f_h
        + w * stride_f_w
    )
    val = tl.load(inp_ptr, mask=mask, other=0.0)

    # write to output (B, M, C) which is contiguous in C dimension
    out_ptr = out_ptr + b * (M * C) + m * C + offs
    tl.store(out_ptr, val, mask=mask)


def triton_gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """
    Implements _transpose_and_gather_feat using a single Triton kernel.
    Returns tensor of shape (B, M, C).
    """
    assert feat.is_cuda and ind.is_cuda
    feat = feat.contiguous()
    ind = ind.contiguous()

    B, C, H, W = feat.shape
    M = ind.shape[1]

    out = torch.empty((B, M, C), dtype=feat.dtype, device=feat.device)

    BLOCK_SIZE = 128
    grid = (
        (B * M, (C + BLOCK_SIZE - 1) // BLOCK_SIZE)
    )  # (pid0, pid1)

    gather_kernel[grid](
        feat,
        ind,
        out,
        B,
        C,
        H,
        W,
        M,
        feat.stride(0),
        feat.stride(1),
        feat.stride(2),
        feat.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def smooth_l1_kernel(
    pred_ptr,      # *float32, (B, M, C)
    target_ptr,    # *float32, (B, M, C)
    mask_ptr,      # *float32, (B, M)
    loss_ptr,      # *float32, (B, M, C)
    B, M, C,
    stride_p_bc,   # stride batch in pred
    stride_p_m,    # stride M in pred
    stride_p_c,    # stride C in pred
    stride_t_bc,
    stride_t_m,
    stride_t_c,
    stride_m_bc,   # stride batch in mask
    stride_m_m,    # stride M in mask
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * M * C
    mask = offs < total

    # decompose linear index to (b, m, c)
    tmp = offs
    c = tmp % C
    tmp = tmp // C
    m = tmp % M
    b = tmp // M

    # pointers
    pred_off = b * stride_p_bc + m * stride_p_m + c * stride_p_c
    tgt_off = b * stride_t_bc + m * stride_t_m + c * stride_t_c
    mask_off = b * stride_m_bc + m * stride_m_m

    pred = tl.load(pred_ptr + pred_off, mask=mask, other=0.0)
    tgt = tl.load(target_ptr + tgt_off, mask=mask, other=0.0)
    msk = tl.load(mask_ptr + mask_off, mask=mask, other=0.0)  # broadcasted

    diff = (pred - tgt) * msk
    abs_diff = tl.abs(diff)
    loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)

    tl.store(loss_ptr + pred_off, loss, mask=mask)


def triton_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Element‑wise smooth L1 loss with masking.
    Returns a tensor of shape (B, M, C) containing per‑element loss.
    """
    assert pred.is_cuda and target.is_cuda and mask.is_cuda
    pred = pred.contiguous()
    target = target.contiguous()
    mask = mask.contiguous().float()  # ensure float for multiplication

    B, M, C = pred.shape
    loss = torch.empty_like(pred)

    BLOCK_SIZE = 256
    grid = ( (B * M * C + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    smooth_l1_kernel[grid](
        pred,
        target,
        mask,
        loss,
        B,
        M,
        C,
        pred.stride(0),
        pred.stride(1),
        pred.stride(2),
        target.stride(0),
        target.stride(1),
        target.stride(2),
        mask.stride(0),
        mask.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return loss


# -------------------- Optimized model -------------------- #

class ModelNew(nn.Module):
    """Regression loss with Triton‑accelerated gather and smooth‑L1."""

    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, mask: torch.Tensor,
                ind: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: (B, C, H, W)   – raw network output
            mask:   (B, M)          – 1/0 mask of valid objects
            ind:    (B, M)          – flat indices into H*W
            target: (B, M, C)       – regression targets
        Returns:
            scalar loss (torch.float32)
        """
        # 1) fused transpose + gather
        pred = triton_gather(output, ind)          # (B, M, C)

        # 2) masked smooth‑L1 loss
        elem_loss = triton_smooth_l1(pred, target, mask)  # (B, M, C)

        # 3) final reduction (same as original implementation)
        num = mask.float().sum()
        loss = elem_loss.sum() / (num + 1e-4)
        return loss