import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------


@triton.jit
def linear_kernel(
    X_ptr,          # [B, N, Fin] input
    W_ptr,          # [Fin, Fout] weight
    Out_ptr,        # [B, N, Fout] output
    B, N, Fin, Fout,
    BLOCK_N: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid_n = tl.program_id(0)  # block over N dimension
    pid_b = tl.program_id(1)  # block over batch dimension

    n_start = pid_n * BLOCK_N
    b = pid_b

    # offsets for the N dimension
    n_offset = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offset < N

    # pointer arithmetic
    X_batch_stride = N * Fin
    X_row_stride = Fin
    W_row_stride = Fout
    Out_batch_stride = N * Fout
    Out_row_stride = Fout

    # Load one row of X (size Fin) per active thread
    # We'll accumulate over Fin in chunks of BLOCK_F
    for f_start in range(0, Fin, BLOCK_F):
        f_offset = f_start + tl.arange(0, BLOCK_F)
        mask_f = f_offset < Fin

        # Load X block
        X_ptrs = X_ptr + b * X_batch_stride + n_offset[:, None] * X_row_stride + f_offset[None, :]
        X = tl.load(X_ptrs, mask=mask_n[:, None] & mask_f[None, :], other=0.0)

        # Load W block
        W_ptrs = W_ptr + f_offset[:, None] * W_row_stride + tl.arange(0, BLOCK_F)[None, :]
        # Actually we need the whole column of W (Fin x Fout). We load a tile:
        W = tl.load(W_ptr + f_offset[:, None] * W_row_stride + tl.arange(0, BLOCK_F)[None, :],
                    mask=mask_f[:, None], other=0.0)

        # Compute partial product
        # X: (BLOCK_N, BLOCK_F) , W: (BLOCK_F, Fout)
        # result: (BLOCK_N, Fout)
        prod = tl.dot(X, W)  # Triton provides dot for 2‑D x 2‑D

        # Accumulate into output
        out_ptrs = Out_ptr + b * Out_batch_stride + n_offset[:, None] * Out_row_stride + tl.arange(0, BLOCK_F)[None, :]
        out = tl.load(out_ptrs, mask=mask_n[:, None] & mask_f[None, :], other=0.0)
        out = out + prod
        tl.store(out_ptrs, out, mask=mask_n[:, None] & mask_f[None, :])


def triton_linear(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Batched linear projection: out[b, n, :] = X[b, n, :] @ W
    X: (B, N, Fin)
    W: (Fin, Fout)
    """
    assert X.is_cuda and W.is_cuda
    B, N, Fin = X.shape
    Fout = W.shape[1]

    out = torch.empty((B, N, Fout), dtype=X.dtype, device=X.device)

    BLOCK_N = 64
    BLOCK_F = 32

    grid = ( (N + BLOCK_N - 1) // BLOCK_N,
             B, )

    linear_kernel[grid](
        X,
        W,
        out,
        B, N, Fin, Fout,
        BLOCK_N=BLOCK_N,
        BLOCK_F=BLOCK_F,
    )
    return out


@triton.jit
def matvec_kernel(
    X_ptr,        # [B, N, F] input matrix
    v_ptr,        # [F] vector
    out_ptr,      # [B, N] output vector
    B, N, F,
    BLOCK_N: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid_n = tl.program_id(0)   # block over N
    pid_b = tl.program_id(1)   # batch index

    n_start = pid_n * BLOCK_N
    b = pid_b

    n_offset = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offset < N

    X_batch_stride = N * F
    X_row_stride = F
    out_batch_stride = N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for f_start in range(0, F, BLOCK_F):
        f_offset = f_start + tl.arange(0, BLOCK_F)
        mask_f = f_offset < F

        X_ptrs = X_ptr + b * X_batch_stride + n_offset[:, None] * X_row_stride + f_offset[None, :]
        X = tl.load(X_ptrs, mask=mask_n[:, None] & mask_f[None, :], other=0.0)

        v = tl.load(v_ptr + f_offset, mask=mask_f, other=0.0)

        # dot product over F dimension
        acc = acc + tl.sum(X * v, axis=1)

    out_ptrs = out_ptr + b * out_batch_stride + n_offset
    tl.store(out_ptrs, acc, mask=mask_n)


def triton_matvec(X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes y[b, n] = X[b, n, :] @ v
    X: (B, N, F)
    v: (F,)
    """
    assert X.is_cuda and v.is_cuda
    B, N, F = X.shape
    out = torch.empty((B, N), dtype=X.dtype, device=X.device)

    BLOCK_N = 64
    BLOCK_F = 32

    grid = ( (N + BLOCK_N - 1) // BLOCK_N,
             B, )

    matvec_kernel[grid](
        X,
        v,
        out,
        B, N, F,
        BLOCK_N=BLOCK_N,
        BLOCK_F=BLOCK_F,
    )
    return out


@triton.jit
def add_score_kernel(
    left_ptr,   # [B, N]   dot_i
    right_ptr,  # [B, N]   dot_j
    out_ptr,    # [B, N, N] e
    B, N,
    BLOCK_N: tl.constexpr,
):
    pid_i = tl.program_id(0)   # block over i
    pid_j = tl.program_id(1)   # block over j
    pid_b = tl.program_id(2)   # batch

    i_start = pid_i * BLOCK_N
    j_start = pid_j * BLOCK_N
    b = pid_b

    i_offset = i_start + tl.arange(0, BLOCK_N)
    j_offset = j_start + tl.arange(0, BLOCK_N)

    mask_i = i_offset < N
    mask_j = j_offset < N
    mask = mask_i[:, None] & mask_j[None, :]

    # Load left and right vectors
    left = tl.load(left_ptr + b * N + i_offset, mask=mask_i, other=0.0)
    right = tl.load(right_ptr + b * N + j_offset, mask=mask_j, other=0.0)

    # Broadcast and add
    out = left[:, None] + right[None, :]

    # Store result
    out_ptrs = out_ptr + b * N * N + i_offset[:, None] * N + j_offset[None, :]
    tl.store(out_ptrs, out, mask=mask)


def triton_add_score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """
    left/right: (B, N)
    returns e: (B, N, N) where e[b,i,j] = left[b,i] + right[b,j]
    """
    assert left.is_cuda and right.is_cuda
    B, N = left.shape
    out = torch.empty((B, N, N), dtype=left.dtype, device=left.device)

    BLOCK_N = 64
    grid = (
        (N + BLOCK_N - 1) // BLOCK_N,
        (N + BLOCK_N - 1) // BLOCK_N,
        B,
    )
    add_score_kernel[grid](
        left,
        right,
        out,
        B, N,
        BLOCK_N=BLOCK_N,
    )
    return out


# ----------------------------------------------------------------------
# Optimized GraphAttentionLayer using the kernels above
# ----------------------------------------------------------------------


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h:   (B, N, Fin)
        adj: (B, N, N) adjacency (binary)
        """
        # 1) Linear projection (batched matmul)
        Wh = triton_linear(h, self.W)                     # (B, N, Fout)

        # 2) Split attention vector a into left/right parts
        a_l = self.a[:self.out_features, 0]               # (Fout,)
        a_r = self.a[self.out_features:, 0]               # (Fout,)

        # 3) Compute dot products for each node
        left = triton_matvec(Wh, a_l)                     # (B, N)
        right = triton_matvec(Wh, a_r)                    # (B, N)

        # 4) Build attention scores e_ij = LeakyReLU(left_i + right_j)
        e = triton_add_score(left, right)                 # (B, N, N)
        e = self.leakyrelu(e)

        # 5) Mask with adjacency (where adj == 0 we put a large negative)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # 6) Softmax over neighbours
        attention = F.softmax(attention, dim=2)

        # 7) Dropout on attention
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 8) Weighted sum of neighbour features
        h_prime = torch.bmm(attention, Wh)                # (B, N, Fout)

        h_prime = F.leaky_relu(h_prime)
        return h_prime, attention


# The model name expected by the benchmark harness
ModelNew = GraphAttentionLayer