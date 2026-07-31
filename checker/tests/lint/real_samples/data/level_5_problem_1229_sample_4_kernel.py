import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.distributions import RelaxedOneHotCategorical


# ----------------------------------------------------------------------
# Triton kernel: write a one‑hot vector given indices (last‑dim argmax)
# ----------------------------------------------------------------------
@triton.jit
def one_hot_kernel(
    idx_ptr,          # *int32  pointer to indices (N,)
    out_ptr,          # *fp32   pointer to output (N, C)
    n_rows,           # i32     number of rows = N
    n_cols,           # i32     number of columns = C
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)                     # each program processes one row
    mask_row = row < n_rows

    # load the index for this row
    idx = tl.load(idx_ptr + row, mask=mask_row, other=0).to(tl.int32)

    # compute the absolute offset of the element that must be set to 1
    # (row * n_cols) + idx
    offset = row * n_cols + idx

    # store a 1.0 at the computed position
    tl.store(out_ptr + offset, tl.full((1,), 1.0, dtype=tl.float32), mask=mask_row)


def triton_one_hot(indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Wrapper that launches the Triton one‑hot kernel.
    `indices` must be a 1‑D int64 tensor of length N (= prod(shape[:-1])).
    `shape` is the desired output shape (..., C) where C = shape[-1].
    """
    assert indices.is_cuda and indices.dtype == torch.int64
    device = indices.device
    N = indices.numel()
    C = shape[-1]

    out = torch.zeros(shape, dtype=torch.float32, device=device)

    # launch one program per row
    grid = (N,)

    # Triton kernels expect int32 for dimensions
    one_hot_kernel[grid](
        indices,
        out,
        N,
        C,
        BLOCK_SIZE=128,
    )
    return out


# ----------------------------------------------------------------------
# Gumbel‑Softmax layer with Triton‑accelerated one‑hot creation
# ----------------------------------------------------------------------
def gumbel_softmax_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    training: bool = True,
    straight_through: bool = False,
) -> torch.Tensor:
    """
    Same semantics as the original function but the argmax‑+‑scatter
    part is replaced by a fused Triton kernel.
    """
    if not training:
        # inference: deterministic argmax → one‑hot
        indexes = logits.argmax(dim=-1).reshape(-1)
        one_hot = triton_one_hot(indexes, logits.shape)
        return one_hot

    # training: sample from RelaxedOneHotCategorical
    sample = RelaxedOneHotCategorical(logits=logits, temperature=temperature).rsample()

    if straight_through:
        # straight‑through estimator: hard = one‑hot(argmax(sample))
        # but gradients flow as if identity (sample) was used.
        idx = sample.argmax(dim=-1).reshape(-1)
        hard = triton_one_hot(idx, sample.shape)

        # `hard - sample` is detached, so its gradient is zero;
        # the gradient of the whole expression is therefore the gradient of `sample`.
        sample = sample + (hard - sample).detach()

    return sample


class GumbelSoftmaxLayer(nn.Module):
    def __init__(self, temperature: float = 1.0, trainable_temperature: bool = False, straight_through: bool = False):
        super().__init__()
        self.straight_through = straight_through
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = nn.Parameter(torch.tensor([temperature], dtype=torch.float32), requires_grad=True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # `self.temperature` may be a Parameter (tensor) or a plain float
        temp = self.temperature if isinstance(self.temperature, torch.Tensor) else float(self.temperature)
        return gumbel_softmax_sample(logits, temp, self.training, self.straight_through)


# ----------------------------------------------------------------------
# Exported optimized model
# ----------------------------------------------------------------------
ModelNew = GumbelSoftmaxLayer