import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------------------------------------------------
# Triton kernel for element‑wise addition (used for the residual)
# ------------------------------------------------------------
@triton.jit
def add_kernel(
    x_ptr,          # *Pointer* to first input tensor
    y_ptr,          # *Pointer* to second input tensor
    out_ptr,        # *Pointer* to output tensor
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the start index of the block that this program will process
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Offsets inside the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to avoid out‑of‑bounds accesses
    mask = offsets < n_elements

    # Load inputs, compute sum and store result
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper that launches the Triton addition kernel.
    Both tensors must be CUDA, contiguous and have the same shape.
    """
    assert x.is_cuda and y.is_cuda, "Both tensors must be on CUDA"
    assert x.shape == y.shape, "Shapes must match for addition"
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 128  # good default; can be tuned per GPU

    # Compute the number of program instances (blocks)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ------------------------------------------------------------
# Original building blocks (unchanged)
# ------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class ChannelMixer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, expansion_factor, dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        out = x + residual
        return out


# ------------------------------------------------------------
# Optimized model – uses Triton for the residual addition
# ------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Same architecture as ``ChannelMixer`` but the residual addition
    (x + residual) is performed with a custom Triton kernel.
    """
    def __init__(self, d_model: int, expansion_factor: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, expansion_factor, dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # Triton‑based residual addition
        out = triton_add(x, residual)
        return out