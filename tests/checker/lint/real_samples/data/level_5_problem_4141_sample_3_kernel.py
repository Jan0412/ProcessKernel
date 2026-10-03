import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------- Triton kernels ------------------- #
@triton.jit
def elementwise_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    OP: tl.constexpr,          # 0=tanh, 1=sigmoid, 2=mask_mul, 3=dropout
    mask_ptr,
    p_keep,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    if OP == 0:          # tanh
        out = tl.tanh(x)
    elif OP == 1:        # sigmoid
        out = 1.0 / (1.0 + tl.exp(-x))
    elif OP == 2:        # mask multiplication (mask_ptr points to mask float)
        m = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
        out = x * m
    elif OP == 3:        # dropout (p_keep = 1 - p)
        # generate random numbers in [0,1)
        rand = tl.rand(tl.arange(0, BLOCK_SIZE), seed=12345, seq_offset=block_start)
        dropout_mask = (rand < p_keep).to(tl.float32)
        scale = 1.0 / p_keep
        out = x * dropout_mask * scale
    else:
        out = x  # fallback (should never happen)

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_tanh(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 128
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    elementwise_kernel[grid](x, out, n, OP=0, mask_ptr=0, p_keep=0.0, BLOCK_SIZE=BLOCK)
    return out


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 128
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    elementwise_kernel[grid](x, out, n, OP=1, mask_ptr=0, p_keep=0.0, BLOCK_SIZE=BLOCK)
    return out


def triton_mul_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and mask.is_cuda
    x = x.contiguous()
    mask = mask.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 128
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    elementwise_kernel[grid](
        x,
        out,
        n,
        OP=2,
        mask_ptr=mask,
        p_keep=0.0,
        BLOCK_SIZE=BLOCK,
    )
    return out


def triton_dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    assert x.is_cuda
    assert 0.0 <= p < 1.0
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 128
    p_keep = 1.0 - p
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    elementwise_kernel[grid](
        x,
        out,
        n,
        OP=3,
        mask_ptr=0,
        p_keep=p_keep,
        BLOCK_SIZE=BLOCK,
    )
    return out


# ------------------- Optimized Model ------------------- #
class ModelNew(nn.Module):
    """Optimized BART Classification Head using Triton kernels."""

    def __init__(self, input_dim: int, inner_dim: int, pooler_dropout: float):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim, bias=True)
        self.out_proj = nn.Linear(inner_dim, 1, bias=True)
        self.p = pooler_dropout

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        # First dropout (using Triton)
        hidden_states = triton_dropout(hidden_states, self.p)

        # First linear + tanh activation (use PyTorch linear, Triton tanh)
        hidden_states = self.dense(hidden_states)
        hidden_states = triton_tanh(hidden_states)

        # Second dropout
        hidden_states = triton_dropout(hidden_states, self.p)

        # Output projection
        hidden_states = self.out_proj(hidden_states)

        # Sigmoid activation
        hidden_states = triton_sigmoid(hidden_states)

        # squeeze and mask multiplication (mask is float)
        hidden_states = hidden_states.squeeze(-1)
        hidden_states = triton_mul_mask(hidden_states, mask.float())

        return hidden_states