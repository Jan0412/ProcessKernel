import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------------------------------
# Triton kernel that computes the sum of |x-y|
# -------------------------------------------------
@triton.jit
def l1_sum_kernel(
    x_ptr,          # *Pointer* to first input tensor
    y_ptr,          # *Pointer* to second input tensor
    sum_ptr,        # *Pointer* to a single‑element tensor holding the partial sum
    n_elements,     # Total number of elements in the tensors
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    diff = tl.abs(x - y)

    # Reduce the diff values inside the block
    block_sum = tl.sum(diff, axis=0)

    # Atomically add the block sum to the global accumulator
    tl.atomic_add(sum_ptr, block_sum)


def triton_l1_mean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute mean absolute error (L1 loss) between two tensors using a custom Triton kernel.
    """
    assert x.is_cuda and y.is_cuda, "Both tensors must be on CUDA."
    assert x.shape == y.shape, "Input tensors must have the same shape."

    # Ensure contiguous layout for pointer arithmetic
    x = x.contiguous()
    y = y.contiguous()

    n_elements = x.numel()
    BLOCK_SIZE = 1024  # a good default; can be tuned per GPU

    # Allocate a single‑element tensor that will hold the sum of absolute differences
    sum_tensor = torch.zeros(1, dtype=x.dtype, device=x.device)

    # Compute grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    l1_sum_kernel[grid](
        x,
        y,
        sum_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Convert sum to mean
    mean = sum_tensor[0] / n_elements
    return mean


# -------------------------------------------------
# Optimized model that replaces the Python loop + nn.L1Loss
# -------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # No sub‑modules needed; loss is computed by the Triton kernel

    def forward(self, results: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Expects `results` and `targets` to be tensors of shape (B, ...).
        The original implementation iterated over the batch dimension and
        averaged per‑sample L1 losses. This is mathematically identical to
        computing the mean absolute error over the whole tensor, which we
        implement with a single fused Triton kernel.
        """
        return triton_l1_mean(results, targets)


# -------------------------------------------------
# Helper functions matching the original benchmark interface
# -------------------------------------------------
def get_inputs():
    # Same shape as the original benchmark
    return [torch.rand([4, 4, 4, 4], device="cuda"),
            torch.rand([4, 4, 4, 4], device="cuda")]


def get_init_inputs():
    # No extra initialization needed for the optimized model
    return []


# Alias for the benchmark harness
Model = ModelNew