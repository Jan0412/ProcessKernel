import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------
@triton.jit
def fused_linear_kernel(
    x_ptr,            # input X (N, D)
    w_mean_ptr,       # weight for mean (D, D)
    b_mean_ptr,       # bias for mean (D)
    w_sc_ptr,         # weight for scale (D, D)
    b_sc_ptr,         # bias for scale (D)
    out_mean_ptr,     # output mean (N, D)
    out_sc_ptr,       # output scale (N, D)
    N, D,
    stride_x0, stride_x1,
    stride_wm0, stride_wm1,
    stride_ws0, stride_ws1,
    stride_out0, stride_out1,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)          # 0 .. N-1
    col = tl.program_id(1)          # 0 .. D-1

    # mask for the inner dimension (D is tiny, usually 4)
    k = tl.arange(0, BLOCK_D)
    mask = k < D

    # load a row of X
    x = tl.load(x_ptr + row * stride_x0 + k * stride_x1, mask=mask, other=0.0)

    # load corresponding rows of the two weight matrices
    w_mean = tl.load(w_mean_ptr + col * stride_wm0 + k * stride_wm1, mask=mask, other=0.0)
    w_sc   = tl.load(w_sc_ptr   + col * stride_ws0 + k * stride_ws1, mask=mask, other=0.0)

    # dot products
    dot_mean = tl.dot(x, w_mean)
    dot_sc   = tl.dot(x, w_sc)

    # load biases
    b_mean = tl.load(b_mean_ptr + col, mask=col < D, other=0.0)
    b_sc   = tl.load(b_sc_ptr   + col, mask=col < D, other=0.0)

    # write results
    out_mean = dot_mean + b_mean
    out_sc   = dot_sc   + b_sc

    tl.store(out_mean_ptr + row * stride_out0 + col * stride_out1, out_mean)
    tl.store(out_sc_ptr   + row * stride_out0 + col * stride_out1, out_sc)


@triton.jit
def diag_exp_kernel(
    sc_ptr,                 # (N, D) scale values (pre‑exp)
    out_ptr,                # (N, D, D) output matrix
    N, D,
    stride_sc0, stride_sc1,
    stride_out0, stride_out1, stride_out2,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)          # 0 .. N-1
    col = tl.program_id(1)          # 0 .. D-1

    # load the element sc[row, col]
    sc = tl.load(sc_ptr + row * stride_sc0 + col * stride_sc1, mask=col < D, other=0.0)

    # compute exp
    val = tl.exp(sc)

    # store at (row, col, col) – the diagonal position
    tl.store(out_ptr + row * stride_out0 + col * stride_out1 + col * stride_out2, val)


# ----------------------------------------------------------------------
# Helper wrappers
# ----------------------------------------------------------------------
def triton_fused_linear(x: torch.Tensor,
                        w_mean: torch.Tensor, b_mean: torch.Tensor,
                        w_sc: torch.Tensor,   b_sc: torch.Tensor):
    """
    Computes two linear layers in a single Triton kernel.
    Returns (out_mean, out_sc) both shaped (N, D).
    """
    assert x.is_cuda and w_mean.is_cuda
    N, D = x.shape
    out_mean = torch.empty_like(x)
    out_sc   = torch.empty_like(x)

    # strides
    stride_x0, stride_x1 = x.stride()
    stride_wm0, stride_wm1 = w_mean.stride()
    stride_ws0, stride_ws1 = w_sc.stride()
    stride_out0, stride_out1 = out_mean.stride()

    grid = (N, D)

    fused_linear_kernel[grid](
        x,
        w_mean,
        b_mean,
        w_sc,
        b_sc,
        out_mean,
        out_sc,
        N, D,
        stride_x0, stride_x1,
        stride_wm0, stride_wm1,
        stride_ws0, stride_ws1,
        stride_out0, stride_out1,
        BLOCK_D=D,
    )
    return out_mean, out_sc


def triton_diag_exp(sc: torch.Tensor):
    """
    Given (N, D) tensor `sc`, returns (N, D, D) where the diagonal
    contains exp(sc) and the rest is zero.
    """
    N, D = sc.shape
    out = torch.zeros(N, D, D, dtype=sc.dtype, device=sc.device)

    stride_sc0, stride_sc1 = sc.stride()
    stride_out0, stride_out1, stride_out2 = out.stride()

    grid = (N, D)

    diag_exp_kernel[grid](
        sc,
        out,
        N, D,
        stride_sc0, stride_sc1,
        stride_out0, stride_out1, stride_out2,
        BLOCK_D=D,
    )
    return out


# ----------------------------------------------------------------------
# Optimized model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean = nn.Linear(latent_dim, latent_dim, bias=True)
        self.sc   = nn.Linear(latent_dim, latent_dim, bias=True)

    def forward(self, x: torch.Tensor):
        """
        x: arbitrary shape ending with `latent_dim` (e.g., [B, ..., latent_dim])
        Returns:
            mean: same shape as input
            cov: same shape with an extra dim ([..., latent_dim, latent_dim])
                 where each slice is a diagonal covariance matrix.
        """
        orig_shape = x.shape
        D = self.latent_dim
        # Collapse all leading dimensions into a single batch dimension N
        x_flat = x.reshape(-1, D).contiguous()

        # Use our fused Triton kernel for the two linear layers
        mean_flat, sc_flat = triton_fused_linear(
            x_flat,
            self.mean.weight, self.mean.bias,
            self.sc.weight,   self.sc.bias,
        )

        # Reshape mean back to original shape
        mean = mean_flat.view(*orig_shape)

        # Build diagonal covariance matrix from sc
        diag_cov_flat = triton_diag_exp(sc_flat)          # (N, D, D)
        cov = diag_cov_flat.view(*orig_shape[:-1], D, D)  # (..., D, D)

        return mean, cov


# ----------------------------------------------------------------------
# Compatibility shim (the original script expects `Model` to be defined)
# ----------------------------------------------------------------------
Model = ModelNew