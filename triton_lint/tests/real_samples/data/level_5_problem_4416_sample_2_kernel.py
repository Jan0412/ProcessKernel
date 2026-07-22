import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def focal_loss_kernel(
    x_ptr,          # logits
    t_ptr,          # targets
    out_ptr,        # per‑element loss output
    n_elements,     # total number of elements
    alpha,          # scalar alpha
    gamma,          # scalar gamma
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    t = tl.load(t_ptr + offsets, mask=mask)

    # sigmoid
    p = 1.0 / (1.0 + tl.exp(-x))

    # alpha mask (as in the original PyTorch implementation)
    alpha_mask = alpha * t

    # focal loss components (exactly replicating the original formula)
    loss_pos = - tl.pow(1.0 - p, gamma) * tl.log(p) * t * alpha_mask
    loss_neg = - tl.pow(1.0 - p, gamma) * tl.log(1.0 - p) * (1.0 - t) * (1.0 - alpha_mask)

    loss = loss_pos + loss_neg
    tl.store(out_ptr + offsets, loss, mask=mask)


def triton_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    size_average: bool = False,
) -> torch.Tensor:
    """
    Compute focal loss using a fused Triton kernel.
    Returns a scalar loss (sum or mean) matching the original module behavior.
    """
    assert inputs.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    inputs = inputs.contiguous()
    targets = targets.contiguous()

    out = torch.empty_like(inputs)
    n_elements = inputs.numel()
    BLOCK_SIZE = 1024  # reasonable default; can be tuned

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    focal_loss_kernel[grid](
        inputs,
        targets,
        out,
        n_elements,
        alpha,
        gamma,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.mean() if size_average else out.sum()


class ModelNew(nn.Module):
    """
    Triton‑accelerated replacement for the original FocalLossSigmoid module.
    """
    def __init__(self, alpha: float = 0.25, gamma: int = 2, size_average: bool = False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return triton_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            size_average=self.size_average,
        )