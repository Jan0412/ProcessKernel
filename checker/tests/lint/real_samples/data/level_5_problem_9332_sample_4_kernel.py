import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -------------------- Triton kernels -------------------- #

@triton.jit
def matmul_kernel(
    A,          # [M, K]
    B,          # [K, N]
    C,          # [M, N]
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # block start offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # mask for out-of-bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # load A and B tiles
        a = tl.load(
            A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)

    # write back
    tl.store(
        C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication using Triton.
    Supports A: [..., M, K], B: [..., K, N] -> [..., M, N]
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported"
    # fuse batch dimensions
    batch_shape = A.shape[:-2]
    M, K = A.shape[-2:]
    K2, N = B.shape[-2:]
    assert K == K2, "Inner dimensions must match"
    A = A.contiguous()
    B = B.contiguous()
    # output
    C = torch.empty(*batch_shape, M, N, device=A.device, dtype=A.dtype)

    # constants
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # launch grid
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
        int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1,
    )

    # strides
    stride_am, stride_ak = A.stride(-2), A.stride(-1)
    stride_bk, stride_bn = B.stride(-2), B.stride(-1)
    stride_cm, stride_cn = C.stride(-2), C.stride(-1)

    # broadcast batch dimensions via program_id(2)
    # we compute offsets for each batch element
    # Triton currently does not support dynamic batch indexing inside kernel,
    # so we launch a separate kernel per batch using a loop in Python.
    # For simplicity and still good performance on typical batch sizes (<=8),
    # we iterate in Python.
    A_flat = A.view(-1, M, K)
    B_flat = B.view(-1, K, N)
    C_flat = C.view(-1, M, N)

    for b in range(A_flat.shape[0]):
        matmul_kernel[grid](
            A_flat[b],
            B_flat[b],
            C_flat[b],
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            M,
            N,
            K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    return C


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


# -------------------- Optimized Model -------------------- #

class MultiHeadAttentionTriton(nn.Module):
    """Multi-Head Attention with Triton‑accelerated matmuls and residual add."""

    def __init__(self, n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.1,
                 qkv_bias=False, mask_value=0):
        super().__init__()
        self.mask_value = mask_value
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.scale = d_k ** -0.5
        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v
        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)
        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, L_q, _ = q.size()
        _, L_k, _ = k.size()

        # Linear projections
        q = self.linear_q(q).view(B, L_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(B, L_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(B, L_k, self.n_head, self.d_v)

        # Permute to (B, n_head, L, d)
        q = q.permute(0, 2, 1, 3)          # (B, H, L_q, d_k)
        k = k.permute(0, 2, 3, 1)          # (B, H, d_k, L_k)
        v = v.permute(0, 2, 1, 3)          # (B, H, L_k, d_v)

        # ---------- Triton matmul for QK ----------
        logits = triton_matmul(q, k) * self.scale   # (B, H, L_q, L_k)

        # Masking
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)          # (B,1,L_q,L_k)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,L_k)
            logits = logits.masked_fill(mask == self.mask_value,
                                         float("-inf"))

        # Softmax + dropout
        weights = F.softmax(logits, dim=-1)
        weights = self.attn_drop(weights)

        # ---------- Triton matmul for WV ----------
        attn_out = triton_matmul(weights, v)          # (B, H, L_q, d_v)
        attn_out = attn_out.transpose(1, 2)           # (B, L_q, H, d_v)
        attn_out = attn_out.reshape(B, L_q, self.dim_v)

        # Final projection
        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out


class PositionwiseFeedForward(nn.Module):
    """Two‑layer feed‑forward network (unchanged)."""

    def __init__(self, d_in, d_hid, dropout=0.1, act_layer=nn.GELU):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayerNew(nn.Module):
    """Transformer encoder layer using Triton‑based attention."""

    def __init__(self, d_model=512, d_inner=256, n_head=8,
                 d_k=64, d_v=64, dropout=0.1, qkv_bias=False,
                 mask_value=0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionTriton(
            n_head, d_model, d_k, d_v,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_value=mask_value,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(d_model, d_inner,
                                           dropout=dropout,
                                           act_layer=act_layer)

    def forward(self, x, mask=None):
        # First sub‑layer
        residual = x
        x = self.norm1(x)
        x = triton_add(residual, self.attn(x, x, x, mask))

        # Second sub‑layer
        residual = x
        x = self.norm2(x)
        x = triton_add(residual, self.mlp(x))
        return x


# Alias expected by the benchmark harness
ModelNew = TransformerEncoderLayerNew