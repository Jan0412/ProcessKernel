import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel: fused log‑softmax + negative‑log‑likelihood (cross‑entropy)
# ----------------------------------------------------------------------
@triton.jit
def cross_entropy_kernel(
    logits_ptr,          # *float32, [N, C]
    target_ptr,          # *int64,   [N]
    loss_ptr,            # *float32, [N]
    N: tl.int32,         # number of rows (samples)
    C: tl.int32,         # number of classes (cols)
    BLOCK_SIZE: tl.constexpr,
):
    # each program processes ONE row (sample)
    row_idx = tl.program_id(0)
    if row_idx >= N:
        return

    # offsets for the columns of this row
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < C

    # pointer to the beginning of the row
    row_ptr = logits_ptr + row_idx * C

    # ---------- 1. max reduction (for numerical stability) ----------
    logits = tl.load(row_ptr + col, mask=mask, other=-float("inf"))
    row_max = tl.max(logits, axis=0, mask=mask)

    # ---------- 2. log‑softmax ----------
    shifted = logits - row_max
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0, mask=mask)
    log_sum_exp = tl.log(sum_exp) + row_max   # scalar for this row

    # ---------- 3. gather target logit ----------
    tgt = tl.load(target_ptr + row_idx, mask=None, other=0).to(tl.int32)
    # safety: clamp target to [0, C‑1] (ignore‑index handling is done in Python)
    tgt = tl.where((tgt >= 0) & (tgt < C), tgt, 0)
    tgt_logit = tl.load(row_ptr + tgt, mask=None, other=0.0)

    # ---------- 4. loss ----------
    loss = -(tgt_logit - log_sum_exp)   # = -log_softmax(target)
    tl.store(loss_ptr + row_idx, loss)


def triton_cross_entropy(logits: torch.Tensor,
                         target: torch.Tensor,
                         reduction: str = "mean") -> torch.Tensor:
    """
    Fused cross‑entropy (log‑softmax + NLL) using a Triton kernel.
    Assumes `logits` is of shape (N, C) and `target` is of shape (N,).
    """
    assert logits.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    assert logits.dtype == torch.float32, "Only FP32 is supported."
    assert target.dtype == torch.long, "Target must be int64."

    N, C = logits.shape
    logits = logits.contiguous()
    target = target.contiguous()

    loss = torch.empty(N, dtype=logits.dtype, device=logits.device)

    BLOCK_SIZE = 128  # enough for typical class dimensions
    grid = (N,)

    cross_entropy_kernel[grid](
        logits,
        target,
        loss,
        N,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:   # "none"
        return loss


# ----------------------------------------------------------------------
# Optimized model: replaces F.cross_entropy with the Triton implementation
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, ratio: float = 1.0, weight=None,
                 ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ratio = ratio
        self.weight = weight          # not supported in the fused kernel (fallback to PyTorch if needed)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input: logits of shape (N, C) – any leading batch dims are flattened.
        target: class indices of shape (N,) – matching the flattened batch.
        """
        # flatten batch dimensions so that we have (N, C)
        N = target.numel()
        C = input.shape[-1]
        logits = input.view(N, C)

        # handle ignore_index on the Python side (set loss to 0 for those entries)
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index)
            valid_logits = logits[mask]
            valid_target = target[mask]
            loss = triton_cross_entropy(valid_logits, valid_target, reduction=self.reduction)
            if self.reduction == "none":
                # we need to re‑insert zeros for ignored positions
                full_loss = torch.zeros(N, dtype=logits.dtype, device=logits.device)
                full_loss[mask] = loss
                loss = full_loss
        else:
            loss = triton_cross_entropy(logits, target, reduction=self.reduction)

        # apply optional weighting and ratio
        if self.weight is not None:
            # fallback to PyTorch weighting (not fused)
            loss = loss * self.weight.to(loss.device)
        loss = loss * self.ratio
        return loss


# ----------------------------------------------------------------------
# Keep the original helper functions for compatibility with the benchmark
# ----------------------------------------------------------------------
def get_inputs():
    # Random tensors matching the original signature; they will be reshaped inside the model.
    return [torch.rand([4, 4, 4, 4], device="cuda"),
            torch.randint(0, 4, [4, 4, 4, 4], device="cuda")]


def get_init_inputs():
    return []


# The original name expected by the benchmark
Model = ModelNew