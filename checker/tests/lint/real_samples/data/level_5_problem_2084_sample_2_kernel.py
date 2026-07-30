import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------------------------------------------
# Triton kernels
# -------------------------------------------------------------

@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * y, mask=mask)


@triton.jit
def div_mul_kernel(q_ptr, num_ptr, den_ptr, out_ptr,
                   n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0)
    num = tl.load(num_ptr + offsets, mask=mask, other=0.0)
    den = tl.load(den_ptr + offsets, mask=mask, other=1e-12)
    weighted = num / den
    tl.store(out_ptr + offsets, q * weighted, mask=mask)


@triton.jit
def bmm_kernel(A_ptr, B_ptr, C_ptr,
                B, M, K, N,
                stride_a_batch, stride_a_row, stride_a_col,
                stride_b_row, stride_b_col,
                stride_c_batch, stride_c_row, stride_c_col,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Batched matrix multiplication:
        C[b, i, j] = sum_k A[b, i, k] * B[k, j]
    A : (B, M, K)
    B : (K, N)
    C : (B, M, N)
    """
    pid = tl.program_id(0)
    batch = pid // (tl.cdiv(M, BLOCK_M))
    pid_m = pid % (tl.cdiv(M, BLOCK_M))
    pid_n = tl.program_id(1)

    # block coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # mask for out-of-bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = A_ptr + batch * stride_a_batch \
                     + offs_m[:, None] * stride_a_row \
                     + tl.arange(0, BLOCK_K)[None, :] * stride_a_col
    b_ptrs = B_ptr + offs_n[None, :] * stride_b_col \
                     + tl.arange(0, BLOCK_K)[:, None] * stride_b_row

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] & (k + tl.arange(0, BLOCK_K) < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(k + tl.arange(0, BLOCK_K)[:, None] < K) & mask_n[None, :],
                    other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_a_col
        b_ptrs += BLOCK_K * stride_b_row

    c_ptrs = C_ptr + batch * stride_c_batch \
                     + offs_m[:, None] * stride_c_row \
                     + offs_n[None, :] * stride_c_col
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# -------------------------------------------------------------
# Helper wrappers
# -------------------------------------------------------------

def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 128
    grid = lambda meta: ( (n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    sigmoid_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out


def triton_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 128
    grid = lambda meta: ( (n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    mul_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    return out


def triton_div_mul(q: torch.Tensor, num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    assert q.is_cuda and num.is_cuda and den.is_cuda
    q = q.contiguous()
    num = num.contiguous()
    den = den.contiguous()
    out = torch.empty_like(q)
    n = q.numel()
    BLOCK = 128
    grid = lambda meta: ( (n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    div_mul_kernel[grid](q, num, den, out, n, BLOCK_SIZE=BLOCK)
    return out


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A: (B, M, K)
    B: (K, N)
    returns (B, M, N)
    """
    assert A.is_cuda and B.is_cuda
    B_batch, M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((B_batch, M, N), dtype=A.dtype, device=A.device)

    # strides
    stride_a_batch, stride_a_row, stride_a_col = A.stride()
    stride_b_row, stride_b_col = B.stride()
    stride_c_batch, stride_c_row, stride_c_col = C.stride()

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid_m = (B_batch * ((M + BLOCK_M - 1) // BLOCK_M), (N + BLOCK_N - 1) // BLOCK_N)
    bmm_kernel[grid_m](
        A, B, C,
        B_batch, M, K, N,
        stride_a_batch, stride_a_row, stride_a_col,
        stride_b_row, stride_b_col,
        stride_c_batch, stride_c_row, stride_c_col,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


# -------------------------------------------------------------
# Optimized model
# -------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)
        self.project = nn.Linear(hidden_dim, dim, bias=False)
        self.register_buffer(
            "wbias",
            torch.empty(max_seqlen, max_seqlen, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.wbias)

    def forward(self, x):
        """
        x : (B, T, dim)
        """
        B, T, _ = x.shape

        # Linear projections
        Q = self.to_q(x)                     # (B, T, hidden)
        K = self.to_k(x)                     # (B, T, hidden)
        V = self.to_v(x)                     # (B, T, hidden)

        # Prepare bias matrix
        exp_wbias = torch.exp(self.wbias[:T, :T])   # (T, T) on CPU/GPU

        # ---- Triton accelerated ops ----
        # sigmoid(Q)
        Q_sig = triton_sigmoid(Q)

        # exp(K)
        K_exp = torch.exp(K)

        # elementwise K_exp * V
        KV = triton_mul(K_exp, V)                 # (B, T, hidden)

        # numerator = exp_wbias @ KV   (batched matmul)
        num = triton_bmm(KV, exp_wbias)           # (B, T, hidden)

        # denominator = exp_wbias @ K_exp
        den = triton_bmm(K_exp, exp_wbias)        # (B, T, hidden)

        # Y = Q_sig * (num / den)
        Y = triton_div_mul(Q_sig, num, den)       # (B, T, hidden)

        # final linear projection (use PyTorch's efficient implementation)
        out = self.project(Y)

        return out