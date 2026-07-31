import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel: LayerNorm over the last dimension (C) of a (N, L, C) tensor
# ----------------------------------------------------------------------
@triton.jit
def layernorm_kernel(
    x_ptr,                # *const float*  input (N*L, C)
    out_ptr,              # *float*       output (N*L, C)
    weight_ptr,           # *const float* gamma (C)
    bias_ptr,             # *const float* beta  (C)
    n_rows,               # int64         number of rows = N*L
    C,                    # int64         channel dimension
    eps,                  # float32       epsilon
    BLOCK_C: tl.constexpr # compile‑time constant: block size for C
):
    row = tl.program_id(0)                     # each program -> one (N,L) row
    row_mask = row < n_rows

    # offsets for the C dimension
    offs = tl.arange(0, BLOCK_C)
    # mask for valid channels (handles C < BLOCK_C)
    c_mask = offs < C

    # linear offset into the flattened (N*L, C) tensor
    base = row * C
    x = tl.load(x_ptr + base + offs, mask=row_mask & c_mask, other=0.0)

    # ---- compute mean ----
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / C

    # ---- compute variance ----
    diff = x - mean
    sum_sq = tl.sum(diff * diff, axis=0)
    var = sum_sq / C
    inv_std = 1.0 / tl.sqrt(var + eps)

    # ---- normalize ----
    y = (x - mean) * inv_std

    # ---- apply scale (weight) and shift (bias) ----
    w = tl.load(weight_ptr + offs, mask=c_mask, other=0.0)
    b = tl.load(bias_ptr + offs, mask=c_mask, other=0.0)
    y = y * w + b

    # write result
    tl.store(out_ptr + base + offs, y, mask=row_mask & c_mask)


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Triton‑based LayerNorm over the last dimension.
    Expected input shape: (N, L, C)  (C <= 128 for this implementation)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    N, L, C = x.shape
    out = torch.empty_like(x)

    n_rows = N * L
    BLOCK_C = 128                     # maximum supported channel size
    assert C <= BLOCK_C, f"C ({C}) must be <= {BLOCK_C} for the current kernel."

    grid = (n_rows,)
    layernorm_kernel[grid](
        x,
        out,
        weight,
        bias,
        n_rows,
        C,
        eps,
        BLOCK_C=BLOCK_C,
    )
    return out


# ----------------------------------------------------------------------
# Triton‑based LayerNorm module (matches the API of SqueezeBertLayerNorm)
# ----------------------------------------------------------------------
class TritonSqueezeBertLayerNorm(nn.Module):
    """
    LayerNorm that works with NCW layout (N, C, W) by permuting to NWC,
    applying a Triton kernel, and permuting back.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias   = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, W)  ->  (N, W, C)
        x = x.permute(0, 2, 1).contiguous()
        x = triton_layernorm(x, self.weight, self.bias, self.eps)
        # back to (N, C, W)
        return x.permute(0, 2, 1).contiguous()


# ----------------------------------------------------------------------
# Optimised model – Conv + Dropout + Residual + Triton LayerNorm
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Equivalent to the original ConvDropoutLayerNorm but uses a Triton‑based
    LayerNorm implementation for the (C) dimension.
    """
    def __init__(self, cin: int, cout: int, groups: int, dropout_prob: float):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=cin,
            out_channels=cout,
            kernel_size=1,
            groups=groups,
            bias=True,
        )
        self.layernorm = TritonSqueezeBertLayerNorm(cout)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        hidden_states : (N, C_in, L)
        input_tensor  : (N, C_out, L)   (residual connection)
        """
        x = self.conv1d(hidden_states)           # (N, C_out, L)
        x = self.dropout(x)                      # dropout (training only)
        x = x + input_tensor                     # residual add
        x = self.layernorm(x)                    # Triton‑accelerated LayerNorm
        return x


# ----------------------------------------------------------------------
# Helper functions required by the KernelBench harness
# ----------------------------------------------------------------------
def get_inputs():
    # Same signature as the original benchmark harness
    return [torch.rand([4, 4, 4], device="cuda"), torch.rand([4, 4, 4], device="cuda")]


def get_init_inputs():
    # returns the arguments needed to construct ModelNew
    # (cin, cout, groups, dropout_prob)
    return [4, 4, 1, 0.5]