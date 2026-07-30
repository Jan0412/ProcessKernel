import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------
def _launch_fused_linear_relu(x, w, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    assert x.is_cuda and w.is_cuda and b.is_cuda
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = ( (M + BLOCK_M - 1) // BLOCK_M,
             (N + BLOCK_N - 1) // BLOCK_N )

    @triton.jit
    def fused_linear_relu_kernel(
        X_ptr, W_ptr, B_ptr, Out_ptr,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        # accumulator
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # X: (M, K)
            a = tl.load(
                X_ptr + (offs_m[:, None] * K + offs_k[None, :]),
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            # W: (N, K)  we need W^T to do X @ W^T
            b_mat = tl.load(
                W_ptr + (offs_n[None, :] * K + offs_k[:, None]),
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
            acc += tl.dot(a, b_mat, trans_b=True)

        # bias
        bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]

        # ReLU
        acc = tl.maximum(acc, 0.0)

        # store
        tl.store(
            Out_ptr + (offs_m[:, None] * N + offs_n[None, :]),
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    fused_linear_relu_kernel[grid](
        x, w, b, out,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


def _launch_fused_linear_tanh(x, w, b, scale, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    assert x.is_cuda and w.is_cuda and b.is_cuda
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = ( (M + BLOCK_M - 1) // BLOCK_M,
             (N + BLOCK_N - 1) // BLOCK_N )

    @triton.jit
    def fused_linear_tanh_kernel(
        X_ptr, W_ptr, B_ptr, Out_ptr,
        M, N, K,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            a = tl.load(
                X_ptr + (offs_m[:, None] * K + offs_k[None, :]),
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b_mat = tl.load(
                W_ptr + (offs_n[None, :] * K + offs_k[:, None]),
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )
            acc += tl.dot(a, b_mat, trans_b=True)

        bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]

        # tanh then scale
        acc = tl.tanh(acc) * SCALE

        tl.store(
            Out_ptr + (offs_m[:, None] * N + offs_n[None, :]),
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    fused_linear_tanh_kernel[grid](
        x, w, b, out,
        M, N, K,
        SCALE=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


# ----------------------------------------------------------------------
# Helper modules that use the Triton kernels
# ----------------------------------------------------------------------
class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x.contiguous()
        w = self.weight.contiguous()
        b = self.bias.contiguous()
        return _launch_fused_linear_relu(x, w, b)


class LinearTanh(nn.Module):
    def __init__(self, in_features, out_features, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x.contiguous()
        w = self.weight.contiguous()
        b = self.bias.contiguous()
        return _launch_fused_linear_tanh(x, w, b, self.scale)


# ----------------------------------------------------------------------
# Optimized VAE model using the custom Triton kernels
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action):
        super().__init__()
        # Encoder
        self.e1 = LinearReLU(state_dim + action_dim, 750)
        self.e2 = LinearReLU(750, 750)
        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        # Decoder (note the scaling inside LinearTanh)
        self.d1 = LinearReLU(state_dim + latent_dim, 750)
        self.d2 = LinearReLU(750, 750)
        self.d3 = LinearTanh(750, action_dim, scale=max_action)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        # Encoder
        z = F.relu(self.e1(torch.cat([state, action], dim=1)))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)

        # Reparameterization trick
        z = mean + std * torch.randn_like(std)

        # Decoder
        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim),
                            device=state.device, dtype=state.dtype).clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([state, z], dim=1)))
        a = F.relu(self.d2(a))
        # d3 already includes tanh and max_action scaling
        return self.d3(a)


# ----------------------------------------------------------------------
# Helper functions required by the benchmark harness
# ----------------------------------------------------------------------
def get_inputs():
    # match the original signature: returns two tensors (state, action)
    return [torch.rand([4, 4], device='cuda'), torch.rand([4, 4], device='cuda')]


def get_init_inputs():
    # returns the arguments needed to instantiate ModelNew
    return [4, 4, 4, 4]   # state_dim, action_dim, latent_dim, max_action