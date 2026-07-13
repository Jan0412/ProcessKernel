import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -------------------------------------------------------------
# Triton kernel that fuses softmax, weighted sum and variance
# -------------------------------------------------------------
@triton.jit
def attentive_pooling_kernel(
    feature_ptr,          # *[B, T, H] feature tensor
    logits_ptr,           # *[B, T]     attention logits (already with mask added)
    out_ptr,              # *[B, 2*H]   output: [utter_rep, variance] concatenated
    B, T, H,              # dimensions
    eps,                  # small constant for numerical stability
    BLOCK_T: tl.constexpr,  # block size along T dimension
):
    # each program processes one (batch, hidden) pair
    b = tl.program_id(0)          # batch index
    h = tl.program_id(1)          # hidden index

    # offsets for the T dimension
    offs = tl.arange(0, BLOCK_T)
    mask = offs < T

    # pointers to the current (b, :, h) slice and (b, :) logits slice
    feat_ptr = feature_ptr + b * T * H + offs * H + h
    logit_ptr = logits_ptr + b * T + offs

    # load data
    feat = tl.load(feat_ptr, mask=mask, other=0.0)          # [BLOCK_T]
    logits = tl.load(logit_ptr, mask=mask, other=-float('inf'))  # [BLOCK_T]

    # ---------- softmax ----------
    max_logit = tl.max(logits, axis=0)                     # scalar max over T
    logits = tl.where(mask, logits - max_logit, 0.0)
    exp_logits = tl.exp(logits) * mask.to(tl.float32)
    sum_exp = tl.sum(exp_logits, axis=0) + 1e-9
    weights = exp_logits / sum_exp                         # [BLOCK_T]

    # ---------- weighted sum ----------
    utter = tl.sum(weights * feat, axis=0)                 # scalar
    # ---------- weighted sum of squares ----------
    second = tl.sum(weights * feat * feat, axis=0)        # scalar

    # ---------- variance ----------
    var = tl.sqrt(second - utter * utter + eps)

    # store results
    out_utt_ptr = out_ptr + b * (2 * H) + h
    out_var_ptr = out_ptr + b * (2 * H) + H + h
    tl.store(out_utt_ptr, utter)
    tl.store(out_var_ptr, var)


def triton_attentive_pooling(feature: torch.Tensor, logits: torch.Tensor, eps: float = 1e-8):
    """
    Wrapper that launches the fused Triton kernel.
    feature: (B, T, H) float32 CUDA tensor
    logits : (B, T)    float32 CUDA tensor (mask already added)
    Returns (utter_rep, variance) each of shape (B, H)
    """
    assert feature.is_cuda and logits.is_cuda
    B, T, H = feature.shape
    out = torch.empty((B, 2 * H), dtype=feature.dtype, device=feature.device)

    BLOCK_T = 128  # can be tuned
    grid = (B, H)  # 2‑D grid: batch dim, hidden dim
    attentive_pooling_kernel[grid](
        feature,
        logits,
        out,
        B,
        T,
        H,
        eps,
        BLOCK_T=BLOCK_T,
    )
    utter = out[:, :H]
    var = out[:, H:]
    return utter, var


# -------------------------------------------------------------
# Re‑implemented AttentivePooling that uses the Triton kernel
# -------------------------------------------------------------
class AttentivePoolingTriton(nn.Module):
    """
    Attentive pooling where the softmax + weighted‑sum + variance
    are computed in a single Triton kernel.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.W_a = nn.Linear(input_dim, input_dim, bias=True)
        self.W = nn.Linear(input_dim, 1, bias=True)
        self.act_fn = nn.ReLU()

    def forward(self, batch_rep: torch.Tensor, att_mask: torch.Tensor):
        """
        batch_rep : (B, T, H)
        att_mask  : (B, T)   (logits, not yet softmax)
        Returns:
            utter_rep : (B, H)
            variance  : (B, H)
        """
        # linear + ReLU + linear -> (B, T, 1)
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)  # (B, T)
        # add external mask (logits)
        att_logits = att_logits + att_mask

        # fused Triton kernel computes softmax, weighted sum and variance
        utter, var = triton_attentive_pooling(batch_rep, att_logits, eps=1e-8)
        return utter, var


# -------------------------------------------------------------
# ASP module using the Triton‑based attentive pooling
# -------------------------------------------------------------
class ASP(nn.Module):
    """Attentive Statistic Pooling module incorporating attention mask"""

    def __init__(self, out_dim, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, out_dim, bias=True)
        self.ap_layer = AttentivePoolingTriton(out_dim)

    def forward(self, feature_BxTxH: torch.Tensor, att_mask_BxT: torch.Tensor):
        """
        feature_BxTxH : (B, T, H_in)
        att_mask_BxT  : (B, T)   (logits)
        Returns:
            statistic_pooling : (B, 2 * out_dim)
        """
        # linear projection first
        feature = self.linear(feature_BxTxH)            # (B, T, out_dim)

        # Triton‑fused attentive pooling
        sap_vec, variance = self.ap_layer(feature, att_mask_BxT)

        # concatenate mean and std‑dev
        statistic_pooling = torch.cat([sap_vec, variance], dim=-1)  # (B, 2*out_dim)
        return statistic_pooling


# -------------------------------------------------------------
# Exported optimized model
# -------------------------------------------------------------
ModelNew = ASP