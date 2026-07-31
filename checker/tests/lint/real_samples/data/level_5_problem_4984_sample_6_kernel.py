import torch
import triton
import triton.language as tl
import torch.nn as nn

# --------------------------------------------------------------
# Triton kernel that computes Stable BCE loss and atomically
# accumulates the sum of all elements.
# --------------------------------------------------------------
@triton.jit
def stable_bce_kernel(
    inp_ptr,          # *float32   input tensor
    tgt_ptr,          # *float32   target tensor
    out_sum_ptr,      # *float32   scalar accumulator (size 1)
    N,                # i32        total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load inputs
    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    tgt = tl.load(tgt_ptr + offsets, mask=mask, other=0.0)

    # ---- element‑wise loss computation ----
    # neg_abs = -abs(inp)
    neg_abs = -tl.abs(inp)

    # clamp min 0  (equivalent to max(inp, 0))
    clamped = tl.where(inp > 0.0, inp, 0.0)

    # term2 = inp * tgt
    term2 = inp * tgt

    # term3 = log1p(exp(neg_abs))   == (1 + exp(neg_abs)).log()
    term3 = tl.log1p(tl.exp(neg_abs))

    loss = clamped - term2 + term3

    # ---- block reduction ----
    block_sum = tl.sum(loss, axis=0)

    # Atomically add block sum to the global accumulator
    tl.atomic_add(out_sum_ptr, block_sum)


def triton_stable_bce(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Wrapper around the Triton kernel that returns the mean BCE loss.
    """
    assert input.is_cuda and target.is_cuda, "Tensors must be CUDA tensors"
    # Ensure contiguous layout
    input = input.contiguous()
    target = target.contiguous()

    N = input.numel()
    BLOCK_SIZE = 1024  # good trade‑off for most sizes; Triton will split as needed
    grid = lambda meta: ( (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )

    # Allocate a single‑element tensor for the atomic accumulator
    out_sum = torch.zeros(1, dtype=torch.float32, device=input.device)

    # Launch kernel
    stable_bce_kernel[grid](
        input,
        target,
        out_sum,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Compute mean from the accumulated sum
    mean = out_sum / N
    return mean.squeeze()  # return a 0‑dim tensor


# --------------------------------------------------------------
# Optimized model that uses the Triton implementation.
# --------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return triton_stable_bce(input, target)


# Preserve original name for compatibility with the benchmark harness
Model = ModelNew