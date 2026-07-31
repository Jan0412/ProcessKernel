import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------- Triton kernels ---------------------------

@triton.jit
def linear_kernel(
    a_ptr,          # [M, K] input matrix
    w_ptr,          # [K, N] weight matrix
    b_ptr,          # [N] bias
    out_ptr,        # [M, N] output matrix
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute start offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Mask out-of-bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers with offsets
    a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
    w_ptrs = w_ptr + (offs_k[:, None] * N + offs_n[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        cur_k = tl.where(k + offs_k < K, k + offs_k, 0)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & (cur_k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(cur_k[:, None] < K) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, w)
        # advance pointers
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * N

    # Add bias
    if b_ptr != 0:
        b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += b[None, :]

    # Write back
    out_ptrs = out_ptr + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """Linear layer implemented with a Triton GEMM kernel."""
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    else:
        bias = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    M, K = x.shape
    K_w, N = weight.shape
    assert K == K_w

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Tunable block sizes (these work well for many GPUs)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )
    linear_kernel[grid](
        x,
        weight,
        bias if bias.numel() > 1 else 0,
        out,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


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
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out


@triton.jit
def tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.tanh(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_tanh(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out


# --------------------------- Optimized Model ---------------------------

class ModelNew(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # keep parameters as regular nn.Parameter so they are registered
        self.fc1_weight = nn.Parameter(
            torch.randn(2 * 2 * 512, latent_dim, dtype=torch.float32).cuda()
        )
        self.fc1_bias = nn.Parameter(torch.zeros(2 * 2 * 512, dtype=torch.float32).cuda())

        # Convolution‑transpose layers stay as PyTorch implementations (they are already fast)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=1,
                                       padding=1, output_padding=0)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,
                                       padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                                       padding=1, output_padding=0)
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2,
                                       padding=3)

    def forward(self, input):
        # ----- Linear (fc1) with Triton -----
        # input shape: [B, latent_dim]
        x = triton_linear(input, self.fc1_weight.t(), self.fc1_bias)   # weight transposed to [latent, out]
        # reshape to feature map
        x = x.view(x.size(0), 512, 2, 2)

        # ----- ConvTranspose + fused ReLU (using Triton ReLU) -----
        x = triton_relu(self.conv1(x))
        x = triton_relu(self.conv2(x))
        x = triton_relu(self.conv3(x))
        x = triton_relu(self.conv4(x))

        # ----- final tanh -----
        x = triton_tanh(x)
        return x


# --------------------------- Helper for interface ---------------------------

def get_inputs():
    return [torch.rand([4, 4], device='cuda')]

def get_init_inputs():
    # latent_dim = 4 as in the original example
    return [4]

# expose the model class with the expected name for the harness
Model = ModelNew