import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def pointwise_conv_1d_kernel(
    X_ptr, W_ptr, Bias_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    # M = B * H * W
    # N = out_channels
    # K = in_channels
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K
    for start_k in range(0, K, BLOCK_K):
        # Load X
        x_mask = (offs_m[:, None] < M) & ((start_k + offs_k[None, :]) < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load W
        w_mask = (offs_n[:, None] < N) & ((start_k + offs_k[None, :]) < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Dot Product
        acc += tl.dot(x, w.T)
        
        # Update pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        
    # Add Bias if needed
    if HAS_BIAS:
        bias_mask = offs_n < N
        bias = tl.load(Bias_ptr + offs_n, mask=bias_mask, other=0.0)
        acc = acc + bias[None, :]
        
    # Store Y
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)

def triton_pointwise_conv_1d(x, weight, bias=None):
    # x: [B, C, H, W]
    # weight: [O, C, 1, 1]
    
    B, C, H, W = x.shape
    O, _, _, _ = weight.shape
    
    # Flatten spatial dimensions for GEMM
    M = B * H * W
    N = O
    K = C
    
    # Reshape inputs for GEMM
    X = x.view(M, K)
    W = weight.view(N, K)
    
    # Output
    Y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    # Strides
    stride_xm, stride_xk = X.stride()
    stride_wn, stride_wk = W.stride()
    stride_ym, stride_yn = Y.stride()
    
    # Grid
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    
    pointwise_conv_1d_kernel[grid](
        X, W, bias, Y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        HAS_BIAS=bias is not None
    )
    
    # Reshape back
    return Y.view(B, O, H, W)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ModelNew, self).__init__()
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_pointwise_conv_1d(x, self.weight, self.bias)