import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------

@triton.jit
def gemm_bias_kernel(
    A,          # [M, K]
    B,          # [N, K] (we store B transposed)
    C,          # [M, N] output
    bias,       # [N]   (broadcasted along M)
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute start offsets of the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create a mask for out‑of‑bounds rows / cols
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A and B tiles
        a = tl.load(A + (offs_m[:, None] * K + offs_k[None, :]),
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0)
        b = tl.load(B + (offs_n[:, None] * K + offs_k[None, :]),
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0.0)

        # Compute block‑matrix multiplication
        acc += tl.dot(a, b)

    # Add bias (broadcasted on M axis)
    bias_vec = tl.load(bias + offs_n, mask=mask_n, other=0.0)
    acc += bias_vec[None, :]

    # Write back
    tl.store(C + (offs_m[:, None] * N + offs_n[None, :]),
             acc,
             mask=mask_m[:, None] & mask_n[None, :])


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Fully‑connected layer implemented as GEMM + bias.
    x:   (..., in_features)  FP32 CUDA tensor
    weight: (out_features, in_features)  FP32 CUDA tensor
    bias:   (out_features)               FP32 CUDA tensor
    Returns: (..., out_features) tensor
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # flatten leading dimensions
    *lead, K = x.shape
    M = int(torch.prod(torch.tensor(lead)).item())
    N = weight.shape[0]
    x_flat = x.view(M, K)
    out_flat = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # kernel launch parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_bias_kernel[grid](
        x_flat,
        weight,          # note: we pass weight as [N, K] (already transposed)
        out_flat,
        bias,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out_flat.view(*lead, N)


@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu(x: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out


@triton.jit
def softplus_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Softplus: log1p(exp(x))
    out = tl.log1p(tl.exp(x))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softplus(x: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    softplus_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out


# ---------------------------------------------------------
# Optimized model
# ---------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Store weights / biases as parameters (same shape as nn.Linear)
        self.w1 = nn.Parameter(torch.empty(hidden_dim, input_dim, device='cuda'))
        self.b1 = nn.Parameter(torch.empty(hidden_dim, device='cuda'))

        self.w2 = nn.Parameter(torch.empty(hidden_dim, hidden_dim, device='cuda'))
        self.b2 = nn.Parameter(torch.empty(hidden_dim, device='cuda'))

        self.w_loc = nn.Parameter(torch.empty(output_dim, hidden_dim, device='cuda'))
        self.b_loc = nn.Parameter(torch.empty(output_dim, device='cuda'))

        self.w_scale = nn.Parameter(torch.empty(output_dim, hidden_dim, device='cuda'))
        self.b_scale = nn.Parameter(torch.empty(output_dim, device='cuda'))

        # Initialise like nn.Linear would
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b1, -bound, bound)

        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w2)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b2, -bound, bound)

        nn.init.kaiming_uniform_(self.w_loc, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_loc)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b_loc, -bound, bound)

        nn.init.kaiming_uniform_(self.w_scale, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_scale)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b_scale, -bound, bound)

    def forward(self, x):
        # x shape: (B, *, input_dim)
        # Layer 1 + ReLU
        h = triton_linear(x, self.w1, self.b1)
        h = triton_relu(h)

        # Layer 2 + ReLU
        h = triton_linear(h, self.w2, self.b2)
        h = triton_relu(h)

        # Output location (linear only)
        loc = triton_linear(h, self.w_loc, self.b_loc)

        # Output scale (linear + softplus)
        scale = triton_linear(h, self.w_scale, self.b_scale)
        scale = triton_softplus(scale)

        return loc, scale


# ---------------------------------------------------------
# Helper for external scripts (same signature as original)
# ---------------------------------------------------------
def get_init_inputs():
    # (input_dim, hidden_dim, output_dim)
    return [4, 4, 4]

def get_inputs():
    return [torch.rand([4, 4, 4, 4], device='cuda')]