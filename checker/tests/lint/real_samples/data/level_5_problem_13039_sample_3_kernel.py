import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------------------------------------------------
# Triton kernel: fused computation of attention scores
# ------------------------------------------------------------
@triton.jit
def attn_score_kernel(
    x_ptr,          # ptr to input tensor (B, L, F)
    w_i_ptr,        # weight slice for x_i      (F,)
    w_j_ptr,        # weight slice for x_j      (F,)
    w_ij_ptr,       # weight slice for elementwise product (F,)
    out_ptr,        # ptr to output matrix (B, L, L)
    B, L, F,
    BLOCK_J: tl.constexpr,
):
    # program id 0 -> (b, i), program id 1 -> block of j
    pid0 = tl.program_id(0)
    b = pid0 // L
    i = pid0 % L

    # offsets for the feature dimension
    offs_f = tl.arange(0, F)

    # ------------------------------------------------------------------
    # Load x_i and compute the constant part a_i = w_i · x_i
    # ------------------------------------------------------------------
    x_i_ptr = x_ptr + (b * L + i) * F
    x_i = tl.load(x_i_ptr + offs_f, mask=True, other=0.0)
    a_i = tl.sum(x_i * w_i_ptr, axis=0)

    # v_i = x_i * w_ij   (used for the interaction term)
    v_i = x_i * w_ij_ptr

    # ------------------------------------------------------------------
    # Loop over j in blocks of size BLOCK_J
    # ------------------------------------------------------------------
    pid1 = tl.program_id(1)
    j_start = pid1 * BLOCK_J
    j_end = j_start + BLOCK_J
    mask_j = j_start + tl.arange(0, BLOCK_J) < L
    offs_j = j_start + tl.arange(0, BLOCK_J)

    # pointer to the start of the j‑block for this batch
    x_j_base = x_ptr + (b * L) * F

    # Load x_j: shape (BLOCK_J, F)
    x_j = tl.load(x_j_base + offs_j[:, None] * F + offs_f[None, :],
                  mask=mask_j[:, None], other=0.0)

    # b_j = w_j · x_j   (shape BLOCK_J)
    b_j = tl.sum(x_j * w_j_ptr[None, :], axis=1)

    # c_ij = v_i · x_j  (shape BLOCK_J)
    c_ij = tl.sum(v_i[None, :] * x_j, axis=1)

    # score = a_i + b_j + c_ij
    score = a_i + b_j + c_ij

    # Store into the output matrix at (b, i, j_start:j_end)
    out_ptr_base = out_ptr + (b * L + i) * L
    tl.store(out_ptr_base + offs_j, score, mask=mask_j)


def triton_attn_score(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute the (B, L, L) attention score matrix using a fused Triton kernel.
    `weight` must be a 1‑D tensor of size 3*F (the same as the original
    `nn.Linear(3*F, 1, bias=False)` weight).
    """
    assert x.is_cuda and weight.is_cuda
    B, L, F = x.shape
    # split the weight vector
    w_i = weight[:F]
    w_j = weight[F:2 * F]
    w_ij = weight[2 * F:]

    out = torch.empty(B, L, L, device=x.device, dtype=x.dtype)

    BLOCK_J = 32  # can be tuned
    grid = (B * L, (L + BLOCK_J - 1) // BLOCK_J)

    attn_score_kernel[grid](
        x,
        w_i,
        w_j,
        w_ij,
        out,
        B,
        L,
        F,
        BLOCK_J=BLOCK_J,
    )
    return out


# ------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------
class ModelNew(nn.Module):
    """
    SemanticComposite with a fused Triton kernel for the attention score
    computation.
    """
    def __init__(self, in_features, dropout_rate: float = 0.0):
        super().__init__()
        self.in_features = in_features
        # keep the original Linear for weight storage (bias=False)
        self.att_linear = nn.Linear(3 * in_features, 1, bias=False)
        self.z_gate = nn.Linear(2 * in_features, in_features, True)
        self.r_gate = nn.Linear(2 * in_features, in_features, True)
        self.f_gate = nn.Linear(2 * in_features, in_features, True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        x: (B, L, F)
        """
        B, L, F = x.shape

        # ---- fused attention score computation ----
        # weight is stored as (1, 3*F) in the Linear layer
        weight_vec = self.att_linear.weight.squeeze(0)  # (3*F,)
        attn_matrix = triton_attn_score(x, weight_vec)  # (B, L, L)

        # softmax over the key dimension (dim=2) and weighted sum
        attn_weight = torch.softmax(attn_matrix, dim=2)
        attn = torch.bmm(attn_weight, x)  # (B, L, F)

        # ---- gate part (unchanged) ----
        x_attn_concat = torch.cat([x, attn], dim=-1)          # (B, L, 2F)
        x_attn_concat = self.dropout(x_attn_concat)

        z = torch.tanh(self.z_gate(x_attn_concat))
        r = torch.sigmoid(self.r_gate(x_attn_concat))
        f = torch.sigmoid(self.f_gate(x_attn_concat))

        encoding = r * x + f * z
        return encoding


# ------------------------------------------------------------
# Helper functions required by the benchmark harness
# ------------------------------------------------------------
def get_inputs():
    # example input matching the original signature
    return [torch.rand([4, 4, 4], device="cuda")]


def get_init_inputs():
    return [[], {"in_features": 4}]