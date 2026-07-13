import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from abc import ABC, abstractmethod
import numpy as np

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def to_array_as(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, np.ndarray):
        return x.detach().cpu().numpy().astype(y.dtype)
    elif isinstance(x, np.ndarray) and isinstance(y, torch.Tensor):
        return torch.tensor(x, device=y.device, dtype=y.dtype)
    else:
        return x

# ----------------------------------------------------------------------
# Triton fused Linear kernel (matmul + bias + optional activation)
# ----------------------------------------------------------------------
@triton.jit
def fused_linear_kernel(
    A,                # (M, K) input
    B,                # (N, K) weight (not transposed)
    bias,             # (N,) bias
    C,                # (M, N) output
    M, N, K,
    activation: tl.constexpr,   # 0:none, 1:relu, 2:tanh
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # offsets for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # mask for out‑of‑bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # allocate accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # loop over K dimension
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

        # accumulate
        acc += tl.dot(a, b)

    # add bias
    bias_val = tl.load(bias + offs_n, mask=mask_n, other=0.0)
    acc += bias_val[None, :]

    # activation
    if activation == 1:          # relu
        acc = tl.maximum(acc, 0.0)
    elif activation == 2:        # tanh
        acc = tl.tanh(acc)

    # write back
    tl.store(C + (offs_m[:, None] * N + offs_n[None, :]),
             acc,
             mask=mask_m[:, None] & mask_n[None, :])

def triton_linear(x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor,
                  activation: str = None) -> torch.Tensor:
    """
    x: (M, K)    weight: (N, K)    bias: (N,)
    returns (M, N)
    activation: None, 'relu', 'tanh'
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # kernel launch configuration
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    act_code = 0
    if activation == 'relu':
        act_code = 1
    elif activation == 'tanh':
        act_code = 2

    fused_linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        activation=act_code,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out

# ----------------------------------------------------------------------
# Triton based Linear module (fused matmul+bias+activation)
# ----------------------------------------------------------------------
class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        # same init as torch.nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return triton_linear(input, self.weight, self.bias, activation=self.activation)

import math

# ----------------------------------------------------------------------
# Optimized VAE model using TritonLinear
# ----------------------------------------------------------------------
class BasePolicy(ABC):
    @abstractmethod
    def policy_infer(self, obs):
        pass

    def get_action(self, obs):
        obs_tensor = torch.tensor(obs,
                                  device=next(self.parameters()).device,
                                  dtype=torch.float32)
        act = to_array_as(self.policy_infer(obs_tensor), obs)
        return act

class ModelNew(nn.Module, BasePolicy):
    def __init__(self, state_dim, action_dim, latent_dim, max_action,
                 hidden_size=750):
        super(ModelNew, self).__init__()
        # encoder
        self.e1 = TritonLinear(state_dim + action_dim, hidden_size, activation='relu')
        self.e2 = TritonLinear(hidden_size, hidden_size, activation='relu')
        self.mean = TritonLinear(hidden_size, latent_dim, activation=None)
        self.log_std = TritonLinear(hidden_size, latent_dim, activation=None)

        # decoder
        self.d1 = TritonLinear(state_dim + latent_dim, hidden_size, activation='relu')
        self.d2 = TritonLinear(hidden_size, hidden_size, activation='relu')
        self.d3 = TritonLinear(hidden_size, action_dim, activation='tanh')  # tanh will be applied then scaled

        self.max_action = max_action
        self.latent_dim = latent_dim
        self._actor = None   # placeholder for external actor if needed

    def forward(self, state, action):
        # Encoder
        z = self.e1(torch.cat([state, action], dim=1))
        z = self.e2(z)
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        # Decoder
        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state, z=None, clip=None, raw=False):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim),
                            device=state.device, dtype=state.dtype)
            if clip is not None:
                z = z.clamp(-clip, clip)

        a = self.d1(torch.cat([state, z], dim=1))
        a = self.d2(a)
        a = self.d3(a)                     # already has tanh activation
        if raw:
            return a
        return self.max_action * a        # max_action scaling

    def policy_infer(self, obs):
        # assumes an external actor network provides mean/std; here we just use decoder
        # For compatibility with original code, we mimic usage of self._actor if set
        if self._actor is None:
            raise RuntimeError("Actor network not set for policy_infer.")
        return self.decode(obs, z=self._actor(obs)[0])

# ----------------------------------------------------------------------
# Helper functions used by the benchmark harness (unchanged)
# ----------------------------------------------------------------------
def get_inputs():
    return [torch.rand([4, 4], device='cuda'), torch.rand([4, 4], device='cuda')]

def get_init_inputs():
    # returns the arguments needed to construct ModelNew
    return [4, 4, 4, 4]

# Model alias expected by the harness
Model = ModelNew