import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel for the forward pass of Mish: y = x * tanh(softplus(x))
# ----------------------------------------------------------------------
@triton.jit
def mish_forward_kernel(
    x_ptr,          # *const float* input
    out_ptr,        # *float* output
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # softplus = log1p(exp(x))
    exp_x = tl.exp(x)
    softplus = tl.log1p(exp_x)

    # tanh(softplus)
    tanh_sp = tl.tanh(softplus)

    # Mish output
    out = x * tanh_sp

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mish(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that launches the Triton Mish kernel.
    """
    assert x.is_cuda, "triton_mish only works on CUDA tensors"
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elem = x_contig.numel()
    BLOCK_SIZE = 1024  # good default, can be tuned

    grid = lambda meta: ((n_elem + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    mish_forward_kernel[grid](x_contig, out, n_elem, BLOCK_SIZE=BLOCK_SIZE)

    return out.view_as(x)


# ----------------------------------------------------------------------
# Autograd Function that uses the Triton kernel for the forward pass
# and a pure‑torch implementation for the backward pass.
# ----------------------------------------------------------------------
class MishTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return triton_mish(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors

        # Re‑use PyTorch ops for the derivative (same formula as the original)
        sp = torch.nn.functional.softplus(x)          # softplus(x)
        tanh_sp = torch.tanh(sp)                      # tanh(softplus(x))
        sigmoid_x = torch.sigmoid(x)                  # sigmoid(x)

        grad = grad_output * (
            tanh_sp
            + x * sigmoid_x * (1.0 - tanh_sp * tanh_sp)
        )
        return grad


# ----------------------------------------------------------------------
# New model that replaces the original Mish autograd Function
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return MishTritonFunction.apply(x)


# ----------------------------------------------------------------------
# Helper functions (kept identical to the original script)
# ----------------------------------------------------------------------
def get_inputs():
    return [torch.rand([4, 4, 4, 4], device="cuda")]


def get_init_inputs():
    return []


# Alias for compatibility with the original benchmark harness
Model = ModelNew