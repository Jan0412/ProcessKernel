import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------------
# Triton kernel for a batched linear (matrix multiply + bias)
# Input: X (N, D), Weight (C, D), Bias (C)
# Output: Y (N, C)
# ------------------------------------------------------------------
@triton.jit
def linear_kernel(
    X_ptr,          # *float32
    W_ptr,          # *float32
    B_ptr,          # *float32
    Y_ptr,          # *float32
    N, D, C,        # int32 scalars
    BLOCK_N: tl.constexpr,
):
    # program id gives the start row this program works on
    row_start = tl.program_id(0) * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_N)
    mask_rows = rows < N

    # iterate over output classes (C is tiny – 4 in the example – so we can unroll)
    for c in range(C):
        # compute dot product of X[row, :] and W[c, :]
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d in range(D):
            x = tl.load(X_ptr + rows * D + d, mask=mask_rows, other=0.0)
            w = tl.load(W_ptr + c * D + d)               # weight is always needed
            acc += x * w
        b = tl.load(B_ptr + c)
        out = acc + b
        tl.store(Y_ptr + rows * C + c, out, mask=mask_rows)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Batched linear layer implemented with Triton.
    x: (N, D) float32 CUDA tensor
    weight: (C, D) float32 CUDA tensor
    bias: (C,) float32 CUDA tensor
    Returns: (N, C) float32 CUDA tensor
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, D = x.shape
    C = weight.shape[0]

    out = torch.empty((N, C), device=x.device, dtype=x.dtype)

    BLOCK_N = 128
    grid = lambda meta: ((N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)

    linear_kernel[grid](
        x,
        weight,
        bias,
        out,
        N,
        D,
        C,
        BLOCK_N=BLOCK_N,
    )
    return out


# ------------------------------------------------------------------
# Triton kernel for Cross‑Entropy (log‑softmax + NLL) loss
# Input: logits (N, C) float32, target (N,) int64
# Output: per‑row loss (N,) float32
# ------------------------------------------------------------------
@triton.jit
def cross_entropy_kernel(
    LOGITS_ptr,    # *float32
    TARGET_ptr,    # *int32
    LOSS_ptr,      # *float32
    N, C,          # int32 scalars
    BLOCK_N: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_N)
    mask_rows = rows < N

    # ---------- max per row ----------
    max_val = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    for c in range(C):
        v = tl.load(LOGITS_ptr + rows * C + c, mask=mask_rows, other=-float("inf"))
        max_val = tl.maximum(max_val, v)

    # ---------- sum of exp(logits - max) ----------
    sum_exp = tl.zeros([BLOCK_N], dtype=tl.float32)
    for c in range(C):
        v = tl.load(LOGITS_ptr + rows * C + c, mask=mask_rows, other=0.0)
        sum_exp += tl.exp(v - max_val)

    log_sum_exp = max_val + tl.log(sum_exp)

    # ---------- gather true‑class logits ----------
    target = tl.load(TARGET_ptr + rows, mask=mask_rows, other=0)
    true_logit = tl.full([BLOCK_N], 0.0, tl.float32)
    for c in range(C):
        is_target = target == c
        v = tl.load(LOGITS_ptr + rows * C + c, mask=mask_rows, other=0.0)
        true_logit = tl.where(is_target, v, true_logit)

    # ---------- loss per row ----------
    loss = log_sum_exp - true_logit
    tl.store(LOSS_ptr + rows, loss, mask=mask_rows)


def triton_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes Cross‑Entropy loss (sum, no reduction) using Triton.
    logits: (N, C) float32 CUDA tensor
    target: (N,) int64 CUDA tensor (class indices)
    Returns: scalar tensor (0‑dim) containing the summed loss (float32)
    """
    assert logits.is_cuda and target.is_cuda
    logits = logits.contiguous()
    target = target.contiguous().to(torch.int32)   # Triton works with int32

    N, C = logits.shape
    per_row_loss = torch.empty(N, device=logits.device, dtype=torch.float32)

    BLOCK_N = 128
    grid = lambda meta: ((N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)
    cross_entropy_kernel[grid](
        logits,
        target,
        per_row_loss,
        N,
        C,
        BLOCK_N=BLOCK_N,
    )
    # sum the per‑row losses (size_average=False -> sum)
    return per_row_loss.sum()


# ------------------------------------------------------------------
# Optimized model using the Triton kernels
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Same functionality as the original SoftmaxLayer but with
    custom Triton kernels for the linear projection and the
    cross‑entropy loss.
    """

    def __init__(self, output_dim: int, n_class: int) -> None:
        super().__init__()
        # keep a regular nn.Linear for parameter handling / initialization
        self.hidden2tag = nn.Linear(output_dim, n_class, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D) where D == output_dim
        y: (...,) containing class indices (int64)
        Returns: scalar loss (sum over all elements)
        """
        # flatten everything except the last feature dimension
        D = x.shape[-1]
        N = x.numel() // D
        x_flat = x.reshape(N, D)

        # custom linear (matmul + bias)
        logits_flat = triton_linear(x_flat, self.hidden2tag.weight, self.hidden2tag.bias)

        # reshape back to original batch shape with new class dimension
        out_shape = list(x.shape[:-1]) + [self.hidden2tag.out_features]
        logits = logits_flat.reshape(out_shape)

        # flatten targets to (N,)
        y_flat = y.reshape(N)

        # custom cross‑entropy (sum reduction)
        loss = triton_cross_entropy(logits, y_flat)
        return loss