import math
import torch
import torch as th
import triton
import triton.language as tl
from torch.nn import Parameter


# --------------------------------------------------------------
# Triton kernel for the 3‑D linear operation (batched per‑channel matmul)
# --------------------------------------------------------------
@triton.jit
def linear3d_kernel(
    # Pointers
    inp_ptr,          # [B, C, K]
    w_ptr,            # [C, K, N]
    bias_ptr,         # [C, N]  (may be nullptr)
    out_ptr,          # [B, C, N]

    # Sizes
    B, C, K, N,

    # Strides (in elements, not bytes)
    stride_inp_b, stride_inp_c, stride_inp_k,
    stride_w_c,  stride_w_k, stride_w_n,
    stride_bias_c, stride_bias_n,
    stride_out_b, stride_out_c, stride_out_n,

    # Compile‑time block sizes
    BLOCK_N: tl.constexpr,
):
    # ------------------------------------------------------------------
    # Program IDs:
    #   pid0 -> batch index   (0 .. B-1)
    #   pid1 -> channel index (0 .. C-1)
    #   pid2 -> output‑feature block
    # ------------------------------------------------------------------
    b = tl.program_id(0)
    c = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Offsets for the N dimension
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # ------------------------------------------------------------------
    # Pointers for this (b, c) slice
    # ------------------------------------------------------------------
    inp_base = inp_ptr + b * stride_inp_b + c * stride_inp_c
    w_base   = w_ptr + c * stride_w_c

    # ------------------------------------------------------------------
    # Accumulator for the output block
    # ------------------------------------------------------------------
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # ------------------------------------------------------------------
    # Loop over K dimension in tiles
    # ------------------------------------------------------------------
    for k in range(0, K, BLOCK_N):
        # NOTE: we reuse BLOCK_N as tile size for K to keep the kernel simple.
        #       For larger K you may want a separate BLOCK_K.
        offs_k = k + tl.arange(0, BLOCK_N)
        mask_k = offs_k < K

        # Load a vector of size [K] from the input
        a = tl.load(inp_base + offs_k * stride_inp_k, mask=mask_k, other=0.0)          # [K]

        # Load a sub‑matrix of size [K, BLOCK_N] from the weight
        w = tl.load(
            w_base
            + offs_k[:, None] * stride_w_k
            + offs_n[None, :] * stride_w_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )                                                                              # [K, BLOCK_N]

        # Compute a @ w  (a is 1‑D, w is 2‑D).  Use a view as a row vector.
        a_row = a[None, :]                     # [1, K]
        acc += tl.dot(a_row, w)[0]             # result -> [BLOCK_N]

    # ------------------------------------------------------------------
    # Add bias if it exists
    # ------------------------------------------------------------------
    if bias_ptr != 0:
        bias_off = bias_ptr + c * stride_bias_c + offs_n * stride_bias_n
        bias = tl.load(bias_off, mask=mask_n, other=0.0)
        acc += bias

    # ------------------------------------------------------------------
    # Write the result
    # ------------------------------------------------------------------
    out_off = out_ptr + b * stride_out_b + c * stride_out_c + offs_n * stride_out_n
    tl.store(out_off, acc, mask=mask_n)


def triton_linear3d(input: torch.Tensor,
                    weight: torch.Tensor,
                    bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    Triton‑accelerated version of ``functional_linear3d``.
    Expected shapes:
        input : (B, C, K)   – float32, contiguous, CUDA
        weight: (C, K, N)   – float32, contiguous, CUDA
        bias  : (C, N)      – float32, contiguous, CUDA or None
    Returns:
        output: (B, C, N)
    """
    assert input.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda

    # Ensure contiguous layout
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C, K = input.shape
    _, _, N = weight.shape

    out = torch.empty((B, C, N), dtype=input.dtype, device=input.device)

    # Compute strides (in elements)
    stride_inp_b, stride_inp_c, stride_inp_k = input.stride()
    stride_w_c, stride_w_k, stride_w_n = weight.stride()
    stride_out_b, stride_out_c, stride_out_n = out.stride()
    if bias is not None:
        stride_bias_c, stride_bias_n = bias.stride()
        bias_ptr = bias
    else:
        # Triton expects a non‑null pointer; we pass 0 and guard inside kernel.
        stride_bias_c = stride_bias_n = 0
        bias_ptr = 0

    # Tunable block size for the N dimension
    BLOCK_N = 128

    grid = (B, C, (N + BLOCK_N - 1) // BLOCK_N)

    linear3d_kernel[grid](
        input,
        weight,
        bias_ptr,
        out,
        B, C, K, N,
        stride_inp_b, stride_inp_c, stride_inp_k,
        stride_w_c, stride_w_k, stride_w_n,
        stride_bias_c, stride_bias_n,
        stride_out_b, stride_out_c, stride_out_n,
        BLOCK_N=BLOCK_N,
    )
    return out


# --------------------------------------------------------------
# Original functional implementation (kept for reference)
# --------------------------------------------------------------
def functional_linear3d(input, weight, bias=None):
    """
    Apply a linear transformation to the incoming data: y = xA^T + b.
    """
    output = input.transpose(0, 1).matmul(weight)
    if bias is not None:
        output += bias.unsqueeze(1)
    return output.transpose(0, 1)


# --------------------------------------------------------------
# Re‑implemented Linear3D that uses the Triton kernel
# --------------------------------------------------------------
class ModelNew(th.nn.Module):
    """
    Same semantics as the original ``Linear3D`` but the core 3‑D linear
    operation is executed by a Triton kernel.
    """

    def __init__(self, channels, in_features, out_features, batch_size=-1,
                 bias=True, noise=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        if noise:
            self.in_features += 1
        self.weight = Parameter(th.Tensor(channels, self.in_features, out_features))
        if bias:
            self.bias = Parameter(th.Tensor(channels, out_features))
        else:
            self.register_parameter('bias', None)
        if noise:
            self.register_buffer('noise', th.Tensor(batch_size, channels, 1))
        self.reset_parameters()
        self.noise_flag = noise

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_matrix=None, permutation_matrix=None):
        """
        Mirrors the original forward logic but delegates the final linear
        transformation to ``triton_linear3d``.
        """
        input_ = [input]

        # ------------------------------------------------------------------
        # 1️⃣  Expand 2‑D inputs to (batch, channels, in_features)
        # ------------------------------------------------------------------
        if input.dim() == 2:
            if permutation_matrix is not None:
                input_.append(
                    input.unsqueeze(1).expand(
                        [input.shape[0], self.channels, permutation_matrix.shape[1]]
                    )
                )
            elif self.noise_flag:
                input_.append(
                    input.unsqueeze(1).expand(
                        [input.shape[0], self.channels, self.in_features - 1]
                    )
                )
            else:
                input_.append(
                    input.unsqueeze(1).expand(
                        [input.shape[0], self.channels, self.in_features]
                    )
                )

        # ------------------------------------------------------------------
        # 2️⃣  Optional adjacency / permutation transformations
        # ------------------------------------------------------------------
        if adj_matrix is not None and permutation_matrix is not None:
            input_.append(
                (input_[-1].transpose(0, 1) @ (adj_matrix.t().unsqueeze(2) *
                 permutation_matrix)).transpose(0, 1)
            )
        elif adj_matrix is not None:
            input_.append(input_[-1] * adj_matrix.t().unsqueeze(0))
        elif permutation_matrix is not None:
            input_.append(
                (input_[-1].transpose(0, 1) @ permutation_matrix).t()
            )

        # ------------------------------------------------------------------
        # 3️⃣  Optional Gaussian noise concatenation
        # ------------------------------------------------------------------
        if self.noise_flag:
            self.noise.normal_()
            input_.append(th.cat([input_[-1], self.noise], dim=2))

        # ------------------------------------------------------------------
        # 4️⃣  Final linear transformation (batched per‑channel)
        # ------------------------------------------------------------------
        final_input = input_[-1]                     # shape (B, C, K)
        return triton_linear3d(final_input, self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def apply_filter(self, permutation_matrix):
        transpose_weight = self.weight.transpose(1, 2) @ permutation_matrix
        self.weight = Parameter(transpose_weight.transpose(1, 2))


# --------------------------------------------------------------
# Helper functions to keep the original API surface
# --------------------------------------------------------------
def get_inputs():
    # Example input generator – matches the original signature
    return [torch.rand([4, 4, 4, 4], device='cuda')]


def get_init_inputs():
    # (channels, in_features, out_features)
    return [4, 4, 4]