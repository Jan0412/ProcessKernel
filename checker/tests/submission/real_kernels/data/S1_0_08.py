import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def get_inputs():
    # Randomly generate input tensors based on the model architecture
    x = torch.rand(16, 64, 512, 512).cuda()
    return [x]


def get_init_inputs():
    # Randomly generate tensors required for initialization based on the model architecture
    return [64, 128, 3, 1, 1, 1]


# -----------------------------------------------------------------------------
# Triton Kernels
# -----------------------------------------------------------------------------

@triton.jit
def depthwise_conv_kernel(
    X, W, BIAS, OUT,
    B, C, H_IN, W_IN, H_OUT, W_OUT,
    KH, KW, STRIDE, DILATION, PAD,
    X_STRIDE_N, X_STRIDE_C, X_STRIDE_H, X_STRIDE_W,
    W_STRIDE_C, W_STRIDE_KH, W_STRIDE_KW,
    OUT_STRIDE_N, OUT_STRIDE_C, OUT_STRIDE_H, OUT_STRIDE_W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Depthwise 2D Convolution Kernel.
    Maps programs to (Batch*Channel, Height, Width).
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Offsets
    offs_c = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_w = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Determine Batch and Channel from offs_c
    # Since we map (B*C) to pid_m, we need to broadcast C and B correctly
    # We assume C is small enough or handled by grid, but here we treat pid_m as index into B*C
    # However, standard Triton MM maps pid_m to rows. Here rows are effectively B*C.
    # Let's simplify: We launch grid over (B, C, H, W) logically.
    # To keep it simple and standard, let's map pid_m to (B, C) flattened.
    # But C is usually fixed per threadblock. Let's assume we launch 1 block per (B, C).
    # Wait, BLOCK_SIZE_M > 1 means we process multiple channels/batches.
    # Let's calculate B and C for every element in offs_c.
    
    # Calculate global batch and channel for each element in the block
    # This is slightly expensive if BLOCK_SIZE_M is large, but necessary for correctness if B*C is flattened.
    # To optimize, we usually assume BLOCK_SIZE_M = 1 for B*C dimension or map it differently.
    # Let's assume BLOCK_SIZE_M = 1 for the B*C dimension to avoid complex modulo arithmetic inside the loop if possible,
    # OR just do the arithmetic.
    
    # B and C indices
    idx_bc = offs_c
    b_idx = idx_bc // C
    c_idx = idx_bc % C

    # Spatial bounds
    mask_h = offs_h < H_OUT
    mask_w = offs_w < W_OUT
    mask_bc = offs_c < (B * C)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    # Loop over Kernel spatial dimensions
    for kh in range(KH):
        for kw in range(KW):
            # Calculate input coordinates
            # h_in = h_out * STRIDE + kh * DILATION - PAD
            h_in = offs_h[:, None, None] * STRIDE + kh * DILATION - PAD
            w_in = offs_w[None, :, None] * STRIDE + kw * DILATION - PAD

            # Load Input X
            # X shape: (B, C, H, W)
            # We need to load X[b_idx, c_idx, h_in, w_in]
            # Broadcast b_idx and c_idx to (BLOCK_SIZE_M, 1, 1)
            b_idx_bc = b_idx[:, None, None]
            c_idx_bc = c_idx[:, None, None]

            # Check bounds for input spatial coordinates
            mask_h_in = (h_in >= 0) & (h_in < H_IN)
            mask_w_in = (w_in >= 0) & (w_in < W_IN)
            mask_in = mask_bc[:, None, None] & mask_h_in & mask_w_in

            # Calculate pointer
            x_ptr = X + b_idx_bc * X_STRIDE_N + c_idx_bc * X_STRIDE_C + h_in * X_STRIDE_H + w_in * X_STRIDE_W
            x_val = tl.load(x_ptr, mask=mask_in, other=0.0)

            # Load Weight W
            # W shape: (C, 1, KH, KW) -> We are at (c_idx, 0, kh, kw)
            # Broadcast c_idx to (BLOCK_SIZE_M, 1, 1)
            # Weight is constant for all h, w in the block for a given c, kh, kw
            w_ptr = W + c_idx_bc * W_STRIDE_C + kh * W_STRIDE_KH + kw * W_STRIDE_KW
            w_val = tl.load(w_ptr, mask=mask_bc[:, None, None], other=0.0)

            acc += x_val * w_val

    # Add Bias if present
    if BIAS is not None:
        bias_val = tl.load(BIAS + c_idx_bc, mask=mask_bc[:, None, None], other=0.0)
        acc += bias_val

    # Store Output
    # OUT shape: (B, C, H, W)
    out_ptr = OUT + b_idx_bc * OUT_STRIDE_N + c_idx_bc * OUT_STRIDE_C + offs_h[:, None, None] * OUT_STRIDE_H + offs_w[None, :, None] * OUT_STRIDE_W
    tl.store(out_ptr, acc, mask=mask_bc[:, None, None] & mask_h[:, None, None] & mask_w[None, :, None])


@triton.jit
def pointwise_conv_kernel(
    X, W, BIAS, OUT,
    B, C_IN, C_OUT, H, W,
    X_STRIDE_N, X_STRIDE_C, X_STRIDE_H, X_STRIDE_W,
    W_STRIDE_COUT, W_STRIDE_CIN,
    OUT_STRIDE_N, OUT_STRIDE_C, OUT_STRIDE_H, OUT_STRIDE_W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Pointwise 2D Convolution (1x1) Kernel.
    Implemented as a GEMM: (B*H*W, C_IN) x (C_IN, C_OUT) -> (B*H*W, C_OUT)
    Maps programs to (Batch*Spatial, Output_Channel).
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Calculate B, H, W indices from offs_m (flattened B*H*W)
    # spatial_dim = H * W
    # b_idx = offs_m // (H * W)
    # hw_idx = offs_m % (H * W)
    # h_idx = hw_idx // W
    # w_idx = hw_idx % W
    
    # To avoid division overhead inside the kernel if possible, we pass B, H, W.
    # Triton can handle simple arithmetic.
    spatial_dim = H * W
    
    b_idx = offs_m // spatial_dim
    hw_idx = offs_m % spatial_dim
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    # Broadcast spatial indices for matrix multiplication logic
    # We are computing dot product for each (m, n) pair.
    # m is (B, H, W), n is C_OUT, k is C_IN
    
    # Load Input X: Shape (B, C_IN, H, W)
    # We need X[b_idx, k, h_idx, w_idx]
    # Broadcast b_idx, h_idx, w_idx to (BLOCK_SIZE_M, 1)
    b_idx_bc = b_idx[:, None]
    h_idx_bc = h_idx[:, None]
    w_idx_bc = w_idx[:, None]
    
    # Load Weight W: Shape (C_OUT, C_IN, 1, 1)
    # We need W[n, k, 0, 0]
    # Broadcast n, k to (1, BLOCK_SIZE_N) and (BLOCK_SIZE_K, 1) respectively?
    # Standard MM: A (M, K) * B (K, N) -> C (M, N)
    # A = X (M, K), B = W (K, N)
    
    # Loop over K (Input Channels)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_start in range(0, C_IN, BLOCK_SIZE_K):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE_K)
        
        # Load X
        x_ptr = X + b_idx_bc * X_STRIDE_N + k_offs[None, :] * X_STRIDE_C + h_idx_bc * X_STRIDE_H + w_idx_bc * X_STRIDE_W
        x_val = tl.load(x_ptr, mask=(offs_m[:, None] < B * spatial_dim) & (k_offs[None, :] < C_IN), other=0.0)
        
        # Load W
        # W is (C_OUT, C_IN, 1, 1)
        # We need W[c_out, k, 0, 0] -> W[n, k]
        w_ptr = W + offs_n[:, None] * W_STRIDE_COUT + k_offs[None, :] * W_STRIDE_CIN
        w_val = tl.load(w_ptr, mask=(offs_n[:, None] < C_OUT) & (k_offs[None, :] < C_IN), other=0.0)
        
        acc += tl.dot(x_val, w_val)

    # Add Bias
    if BIAS is not None:
        bias_val = tl.load(BIAS + offs_n, mask=offs_n < C_OUT, other=0.0)
        acc += bias_val[None, :]

    # Store Output
    # OUT shape: (B, C_OUT, H, W)
    # Map back b_idx, h_idx, w_idx from offs_m
    out_ptr = OUT + b_idx_bc * OUT_STRIDE_N + offs_n[None, :] * OUT_STRIDE_C + h_idx_bc * OUT_STRIDE_H + w_idx_bc * OUT_STRIDE_W
    tl.store(out_ptr, acc, mask=(offs_m[:, None] < B * spatial_dim) & (offs_n[None, :] < C_OUT))


# -----------------------------------------------------------------------------
# Wrapper Functions
# -----------------------------------------------------------------------------

def triton_depthwise2d(x, weight, bias, stride, padding, dilation):
    B, C, Hin, Win = x.shape
    Cout, _, Kh, Kw = weight.shape
    
    Hout = (Hin + 2 * padding - dilation * (Kh - 1) - 1) // stride + 1
    Wout = (Win + 2 * padding - dilation * (Kw - 1) - 1) // stride + 1
    
    out = torch.empty((B, Cout, Hout, Wout), device=x.device, dtype=x.dtype)
    
    # Grid configuration
    # We map (B*C) to M dimension. To keep blocks manageable, let's use small BLOCK_SIZE_M or launch many blocks.
    BLOCK_SIZE_M = 1 # One block per (B, C) to simplify indexing logic in kernel
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    grid = (
        B * C,
        (Hout + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (Wout + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    )
    
    depthwise_conv_kernel[grid](
        x, weight, bias, out,
        B, C, Hin, Win, Hout, Wout,
        Kh, Kw, stride, dilation, padding,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return out


def triton_pointwise2d(x, weight, bias):
    B, Cin, Hin, Win = x.shape
    Cout, _, _, _ = weight.shape
    
    Hout, Wout = Hin, Win
    
    out = torch.empty((B, Cout, Hout, Wout), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid: (B * H * W, Cout, Cin)
    grid = (
        (B * Hin * Win + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (Cout + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (Cin + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    )
    
    pointwise_conv_kernel[grid](
        x, weight, bias, out,
        B, Cin, Cout, Hin, Win,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return out


# -----------------------------------------------------------------------------
# Optimized Model
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Depthwise weights: (in_channels, 1, kernel_size, kernel_size)
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.depthwise_bias = nn.Parameter(torch.zeros(in_channels)) if bias else None

        # Pointwise weights: (out_channels, in_channels, 1, 1)
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        self.pointwise_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution.
        """
        # 1. Depthwise Convolution
        x = triton_depthwise2d(
            x,
            self.depthwise_weight,
            self.depthwise_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

        # 2. Pointwise Convolution (1x1)
        x = triton_pointwise2d(
            x,
            self.pointwise_weight,
            self.pointwise_bias
        )
        return x