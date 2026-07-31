import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np


# ----------------------------------------------------------------------
# Triton kernel: fused attention scores (matmul + interaction term)
# ----------------------------------------------------------------------
@triton.jit
def attn_scores_kernel(
    q_ptr,          # [B, H, S, D]
    k_ptr,          # [B, H, S, D]
    ik_ptr,         # [B, H, S, S, D]
    out_ptr,        # [B, H, S, S]
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_ikb, stride_ikh, stride_iks, stride_ikj, stride_ikd,
    stride_ob, stride_oh, stride_oi, stride_oj,
    BLOCK_D: tl.constexpr,
):
    # program ids
    bh = tl.program_id(0)          # combined batch*head
    i = tl.program_id(1)           # query position
    j = tl.program_id(2)           # key position

    # decode batch and head
    b = bh // H
    h = bh % H

    # offsets
    d_offsets = tl.arange(0, BLOCK_D)

    # accumulate over D
    acc = tl.zeros([1], dtype=tl.float32)

    for d in range(0, D, BLOCK_D):
        cur_D = tl.minimum(BLOCK_D, D - d)
        d_mask = d_offsets < cur_D

        # pointers with offset d
        q_off = q_ptr + b * stride_qb + h * stride_qh + i * stride_qs + (d + d_offsets) * stride_qd
        k_off = k_ptr + b * stride_kb + h * stride_kh + j * stride_ks + (d + d_offsets) * stride_kd
        ik_off = ik_ptr + b * stride_ikb + h * stride_ikh + i * stride_iks + j * stride_ikj + (d + d_offsets) * stride_ikd

        q = tl.load(q_off, mask=d_mask, other=0.0)
        k = tl.load(k_off, mask=d_mask, other=0.0)
        ik = tl.load(ik_off, mask=d_mask, other=0.0)

        # (q * k) + (q * inter_k) summed over d
        prod = q * k + q * ik
        acc += tl.sum(prod, axis=0)

    out_off = out_ptr + b * stride_ob + h * stride_oh + i * stride_oi + j * stride_oj
    tl.store(out_off, acc)


# ----------------------------------------------------------------------
# Triton kernel: fused attention output (scores @ v + interaction term)
# ----------------------------------------------------------------------
@triton.jit
def attn_output_kernel(
    scores_ptr,    # [B, H, S, S]
    v_ptr,        # [B, H, S, D]
    iv_ptr,       # [B, H, S, S, D]
    out_ptr,      # [B, H, S, D]
    B, H, S, D,
    stride_sb, stride_sh, stride_si, stride_sj,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ivb, stride_ivh, stride_ivs, stride_ivj, stride_ivd,
    stride_ob, stride_oh, stride_oi, stride_od,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)          # batch*head
    i = tl.program_id(1)           # query position

    b = bh // H
    h = bh % H

    # accumulate over S (the key/value sequence length)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    s_offsets = tl.arange(0, BLOCK_S)

    for s in range(0, S, BLOCK_S):
        cur_S = tl.minimum(BLOCK_S, S - s)
        s_mask = s_offsets < cur_S

        # pointers
        sc_off = scores_ptr + b * stride_sb + h * stride_sh + i * stride_si + (s + s_offsets) * stride_sj
        v_off = v_ptr + b * stride_vb + h * stride_vh + (s + s_offsets) * stride_vs + tl.arange(0, BLOCK_D) * stride_vd
        iv_off = iv_ptr + b * stride_ivb + h * stride_ivh + i * stride_ivs + (s + s_offsets) * stride_ivj + tl.arange(0, BLOCK_D) * stride_ivd

        scores = tl.load(sc_off, mask=s_mask, other=0.0)          # [cur_S]
        # expand scores to [cur_S, D] for multiplication
        scores = tl.broadcast_to(scores[:, None], (cur_S, BLOCK_D))

        v = tl.load(v_off, mask=s_mask[:, None], other=0.0)      # [cur_S, D]
        iv = tl.load(iv_off, mask=s_mask[:, None], other=0.0)    # [cur_S, D]

        # (scores * v) + (scores * inter_v) summed over s
        acc += tl.sum(scores * v + scores * iv, axis=0)

    out_off = out_ptr + b * stride_ob + h * stride_oh + i * stride_oi + tl.arange(0, BLOCK_D) * stride_od
    tl.store(out_off, acc)


# ----------------------------------------------------------------------
# Helper functions to launch Triton kernels
# ----------------------------------------------------------------------
def triton_attention_scores(q, k, inter_k):
    """
    q: (B, H, S, D)
    k: (B, H, S, D)
    inter_k: (B, H, S, S, D)
    returns scores: (B, H, S, S)
    """
    B, H, S, D = q.shape
    out = torch.empty((B, H, S, S), dtype=q.dtype, device=q.device)

    # strides
    stride_qb, stride_qh, stride_qs, stride_qd = q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = k.stride()
    stride_ikb, stride_ikh, stride_iks, stride_ikj, stride_ikd = inter_k.stride()
    stride_ob, stride_oh, stride_oi, stride_oj = out.stride()

    BLOCK_D = 32

    grid = (B * H, S, S)
    attn_scores_kernel[grid](
        q, k, inter_k, out,
        B, H, S, D,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_ikb, stride_ikh, stride_iks, stride_ikj, stride_ikd,
        stride_ob, stride_oh, stride_oi, stride_oj,
        BLOCK_D=BLOCK_D,
    )
    return out


def triton_attention_output(scores, v, inter_v):
    """
    scores: (B, H, S, S)
    v: (B, H, S, D)
    inter_v: (B, H, S, S, D)
    returns out: (B, H, S, D)
    """
    B, H, S, D = v.shape
    out = torch.empty((B, H, S, D), dtype=v.dtype, device=v.device)

    # strides
    stride_sb, stride_sh, stride_si, stride_sj = scores.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = v.stride()
    stride_ivb, stride_ivh, stride_ivs, stride_ivj, stride_ivd = inter_v.stride()
    stride_ob, stride_oh, stride_oi, stride_od = out.stride()

    BLOCK_S = 32
    BLOCK_D = 32

    grid = (B * H, S)
    attn_output_kernel[grid](
        scores, v, inter_v, out,
        B, H, S, D,
        stride_sb, stride_sh, stride_si, stride_sj,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ivb, stride_ivh, stride_ivs, stride_ivj, stride_ivd,
        stride_ob, stride_oh, stride_oi, stride_od,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )
    return out


# ----------------------------------------------------------------------
# Optimized Model using the Triton kernels
# ----------------------------------------------------------------------
class TimeIntervalTransformerLayerNew(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, kq_same=False):
        super().__init__()
        self.masked_attn_head = TimeIntervalMultiHeadAttention(d_model,
                                                               n_heads,
                                                               kq_same=kq_same)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, inter_k, inter_v, d_k, mask):
        """
        Replace the original implementation with Triton‑based kernels for the
        heavy matrix multiplications and interaction terms.
        """
        # q, k, v : (B, H, S, D)
        # inter_k, inter_v : (B, H, S, S, D)

        # 1) compute attention scores (including interaction)
        scores = triton_attention_scores(q, k, inter_k)          # (B, H, S, S)

        # 2) scale, mask and softmax – keep these in PyTorch (efficient)
        scores = scores / (d_k ** 0.5)
        scores = scores.masked_fill(mask == 0, -np.inf)
        scores = (scores - scores.max(dim=-1, keepdim=True)[0]).softmax(dim=-1)

        # 3) compute output (including interaction)
        out = triton_attention_output(scores, v, inter_v)       # (B, H, S, D)
        return out

    def forward(self, seq, pos_k, pos_v, inter_k, inter_v, mask):
        context = self.masked_attn_head(seq, seq, seq, pos_k, pos_v,
                                        inter_k, inter_v, mask)
        context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output


# Preserve the original MultiHeadAttention class (unchanged)
class TimeIntervalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=False, bias=True):
        super().__init__()
        self.d_model = d_model
        self.h = n_heads
        self.d_k = self.d_model // self.h
        self.kq_same = kq_same
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, pos_k, pos_v, inter_k, inter_v, mask):
        bs, seq_len = k.size(0), k.size(1)
        k = (self.k_linear(k) + pos_k).view(bs, seq_len, self.h, self.d_k)
        if not self.kq_same:
            q = self.q_linear(q).view(bs, seq_len, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, seq_len, self.h, self.d_k)
        v = (self.v_linear(v) + pos_v).view(bs, seq_len, self.h, self.d_k)
        # (B, S, H, D) -> (B, H, S, D)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        inter_k = inter_k.view(bs, seq_len, seq_len, self.h, self.d_k)
        inter_v = inter_v.view(bs, seq_len, seq_len, self.h, self.d_k)
        inter_k = inter_k.transpose(2, 3).transpose(1, 2)   # (B, H, S, S, D)
        inter_v = inter_v.transpose(2, 3).transpose(1, 2)   # (B, H, S, S, D)
        output = self.scaled_dot_product_attention(q, k, v, inter_k,
                                                   inter_v, self.d_k, mask)
        output = output.transpose(1, 2).reshape(bs, -1, self.d_model)
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, inter_k, inter_v, d_k, mask):
        # This staticmethod will be overridden by the subclass (see above)
        raise NotImplementedError("Should be replaced by Triton implementation.")


# Alias expected by the benchmark harness
ModelNew = TimeIntervalTransformerLayerNew