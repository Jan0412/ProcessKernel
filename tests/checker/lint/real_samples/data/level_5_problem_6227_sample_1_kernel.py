import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np


# --------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------

@triton.jit
def linear_kernel(
    x_ptr,          # [M, K] input matrix
    w_ptr,          # [N, K] weight matrix (transposed in matmul)
    b_ptr,          # [N] bias
    out_ptr,        # [M, N] output matrix
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # block start offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # mask for out-of-bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # pointers with offsets
    x_ptrs = x_ptr + (offs_m[:, None] * K + offs_k[None, :])
    w_ptrs = w_ptr + (offs_n[:, None] * K + offs_k[None, :])

    # accumulate
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        cur_k = min(BLOCK_K, K - k)
        k_offset = k + tl.arange(0, cur_k)

        a = tl.load(x_ptrs + k_offset, mask=mask_m[:, None] & (k_offset[None, :] < K), other=0.0)
        b = tl.load(w_ptrs + k_offset, mask=mask_n[:, None] & (k_offset[None, :] < K), other=0.0)

        acc += tl.dot(a, b)

    # add bias
    b_vec = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b_vec[None, :]

    # write back
    out = tl.where(mask_m[:, None] & mask_n[None, :], acc, 0.0)
    tl.store(out_ptr + (offs_m[:, None] * N + offs_n[None, :]), out, mask=mask_m[:, None] & mask_n[None, :])


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: (M, K)
    weight: (N, K)   (i.e. out_features, in_features)
    bias: (N,)
    returns (M, N)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Tunable block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    linear_kernel[grid](
        x,
        weight,
        bias,
        out,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


@triton.jit
def gelu_tanh_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation using tanh (as in the original model)
    sqrt_2_over_pi = tl.constant(np.sqrt(2.0 / np.pi), dtype=tl.float32)
    c = tl.constant(0.044715, dtype=tl.float32)

    y = 0.5 * x * (1.0 + tl.tanh(sqrt_2_over_pi * (x + c * x * x * x)))
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    gelu_tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out


@triton.jit
def dropout_add_kernel(
    x_ptr,    # input
    dropout_ptr,  # dropout mask (0/1)
    out_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    d = tl.load(dropout_ptr + offs, mask=mask, other=0.0)
    out = (x * d) * scale
    tl.store(out_ptr + offs, out, mask=mask)


def triton_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """In‑place dropout returning the scaled tensor (same semantics as nn.Dropout)."""
    if not training or p == 0.0:
        return x
    keep_prob = 1.0 - p
    mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    dropout_add_kernel[grid](x, mask, out, n, scale=1.0 / keep_prob, BLOCK_SIZE=BLOCK)
    return out


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    return out


# --------------------------------------------------------------
# Optimized model (ModelNew)
# --------------------------------------------------------------

class GELU_Triton(nn.Module):
    def forward(self, x):
        return triton_gelu(x)


class Dropout_Triton(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        return triton_dropout(x, self.p, self.training)


class Linear_Triton(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_linear(x, self.weight, self.bias)


class Attention_Triton(nn.Module):
    """Same logic as original Attention but uses Triton‑based linear layers."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = Linear_Triton(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout_Triton(attn_drop)
        self.proj = Linear_Triton(dim, dim)
        self.proj_drop = Dropout_Triton(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)                              # (B, N, 3*dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v)                     # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)        # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp_Triton(nn.Module):
    """MLP using Triton linear + GELU + dropout."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=GELU_Triton, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = Linear_Triton(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = Dropout_Triton(drop)
        self.fc2 = Linear_Triton(hidden_features, out_features)
        self.drop2 = Dropout_Triton(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block_Triton(nn.Module):
    """Block that uses Triton‑accelerated sub‑modules and Triton add for residuals."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=GELU_Triton, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_Triton(dim, num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_Triton(in_features=dim,
                             hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer,
                             drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # Residual + attention
        attn_out = self.attn(self.norm1(x))
        attn_out = self.drop_path(attn_out)
        x = triton_add(x, attn_out)

        # Residual + MLP
        mlp_out = self.mlp(self.norm2(x))
        mlp_out = self.drop_path(mlp_out)
        x = triton_add(x, mlp_out)
        return x


# The name expected by the benchmarking harness
ModelNew = Block_Triton