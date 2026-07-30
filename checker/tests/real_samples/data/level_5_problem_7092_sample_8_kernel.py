import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

import triton
import triton.language as tl


# -------------------- Triton kernels -------------------- #

@triton.jit
def compute_W_kernel(
    w_hat_ptr,  # Parameter W_hat
    m_hat_ptr,  # Parameter M_hat
    out_ptr,    # Output W
    numel,      # total elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    w_hat = tl.load(w_hat_ptr + offs, mask=mask, other=0.0)
    m_hat = tl.load(m_hat_ptr + offs, mask=mask, other=0.0)

    out = tl.tanh(w_hat) * tl.sigmoid(m_hat)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,          # (M, K)
    B_ptr,          # (K, N)
    C_ptr,          # (M, N)
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_offs = offs_m[:, None] * K + tl.arange(0, BLOCK_K)[None, :]
    b_offs = tl.arange(0, BLOCK_K)[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        cur_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + a_offs, mask=(offs_m[:, None] < M) & (cur_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + b_offs, mask=(cur_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_offs = offs_m[:, None] * N + offs_n[None, :]
    tl.store(C_ptr + c_offs,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def combine_kernel(
    a_ptr,      # a tensor
    g_ptr,      # g tensor
    m_ptr,      # m tensor
    out_ptr,    # output y
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0)

    out = g * a + (1.0 - g) * m
    tl.store(out_ptr + offs, out, mask=mask)


# -------------------- Helper wrappers -------------------- #

def triton_compute_W(w_hat: torch.Tensor, m_hat: torch.Tensor) -> torch.Tensor:
    assert w_hat.is_cuda and m_hat.is_cuda
    numel = w_hat.numel()
    out = torch.empty_like(w_hat)

    BLOCK_SIZE = 128
    grid = lambda meta: ((numel + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    compute_W_kernel[grid](w_hat, m_hat, out, numel, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes A @ B where A is (M, K) and B is (K, N) -> (M, N)
    """
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = ( (M + BLOCK_M - 1) // BLOCK_M,
             (N + BLOCK_N - 1) // BLOCK_N )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


def triton_combine(a: torch.Tensor, g: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and g.is_cuda and m.is_cuda
    numel = a.numel()
    out = torch.empty_like(a)

    BLOCK_SIZE = 256
    grid = lambda meta: ((numel + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    combine_kernel[grid](a, g, m, out, numel, BLOCK_SIZE=BLOCK_SIZE)
    return out


# -------------------- Optimized Model -------------------- #

class NeuralAccumulatorCellTriton(nn.Module):
    """NAC cell with Triton‑accelerated linear."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_hat = Parameter(torch.empty(out_dim, in_dim, device='cuda'))
        self.M_hat = Parameter(torch.empty(out_dim, in_dim, device='cuda'))
        self.register_parameter('bias', None)
        self._reset_params()

    def _reset_params(self):
        init.kaiming_uniform_(self.W_hat)
        init.kaiming_uniform_(self.M_hat)

    def forward(self, x: torch.Tensor):
        # x: (..., in_dim)
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_dim)                     # (N, in_dim)

        # compute weight matrix W = tanh(W_hat) * sigmoid(M_hat)
        W = triton_compute_W(self.W_hat, self.M_hat)           # (out_dim, in_dim)

        # linear: x_flat @ W.T  (note B is (out_dim, in_dim) -> we need (in_dim, out_dim))
        out_flat = triton_matmul(x_flat, W.t())                # (N, out_dim)

        return out_flat.reshape(*orig_shape[:-1], self.out_dim)


class ModelNew(nn.Module):
    """NALU cell with Triton kernels."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 1e-10

        self.G = Parameter(torch.empty(out_dim, in_dim, device='cuda'))
        init.kaiming_uniform_(self.G, a=math.sqrt(5))

        self.nac = NeuralAccumulatorCellTriton(in_dim, out_dim)

    def forward(self, input: torch.Tensor):
        # input: (..., in_dim)
        orig_shape = input.shape
        N = input.numel() // self.in_dim                     # total rows after flatten

        # ---- a = NAC(input) ----
        a = self.nac(input)                                 # (..., out_dim)

        # ---- g = sigmoid( linear(input, G) ) ----
        x_flat = input.reshape(N, self.in_dim)              # (N, in_dim)
        g_lin = triton_matmul(x_flat, self.G.t())           # (N, out_dim)
        g = torch.sigmoid(g_lin).reshape(*orig_shape[:-1], self.out_dim)

        # ---- log_input and m = exp( NAC(log_input) ) ----
        log_input = (input.abs() + self.eps).log()
        m_nac = self.nac(log_input)                         # (..., out_dim)
        m = torch.exp(m_nac)

        # ---- combine: y = g * a + (1-g) * m ----
        a_flat = a.reshape(-1)
        g_flat = g.reshape(-1)
        m_flat = m.reshape(-1)
        y_flat = triton_combine(a_flat, g_flat, m_flat)
        y = y_flat.reshape(*orig_shape[:-1], self.out_dim)

        return y