import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    X, # Pointer to input tensor (B, IC, H, W)
    W, # Pointer to weight tensor (OC, IC, KH, KW)
    Bias, # Pointer to bias tensor (OC)
    Out, # Pointer to output tensor (B, OC, OH, OW)
    # Strides for X
    X_STRIDE_B, X_STRIDE_IC, X_STRIDE_H, X_STRIDE_W,
    # Strides for W
    W_STRIDE_OC, W_STRIDE_IC, W_STRIDE_KH, W_STRIDE_KW,
    # Strides for Out
    OUT_STRIDE_B, OUT_STRIDE_OC, OUT_STRIDE_OH, OUT_STRIDE_OW,
    # Dimensions
    B, IC, H, W, OC, KH, KW,
    OH, OW,
    # Hyperparameters
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # Block dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KC: tl.constexpr,
):
    # Determine the batch and output channel handled by this block
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    
    # Determine the spatial tile handled by this block
    pid_m = tl.program_id(2)
    pid_n = tl.program_id(3)

    # Offsets for the output tile
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create masks for valid output positions
    mask_m = off_m < OH
    mask_n = off_n < OW
    mask_2d = mask_m[:, None] & mask_n[None, :]

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over Input Channels
    for start_ic in range(0, IC, BLOCK_KC):
        # Calculate offsets for input channels
        off_ic = start_ic + tl.arange(0, BLOCK_KC)
        mask_ic = off_ic < IC

        # Load Weights for current OC and IC block
        # W shape: (OC, IC, KH, KW)
        # We need W[pid_oc, off_ic, :, :]
        w_offsets = pid_oc * W_STRIDE_OC + off_ic * W_STRIDE_IC
        # We will load weights for all kh, kw later or flatten them?
        # To keep memory coalescing, let's load the full IC block for this OC
        # Weights layout in memory is contiguous for IC, KH, KW (usually)
        
        # We iterate over the kernel area (KH, KW) inside the IC loop to accumulate
        for kh in range(KH):
            for kw in range(KW):
                # Calculate input spatial coordinates corresponding to this kernel position
                # Input H = (Output H * Stride H) + (KH - 1) * Dilation - 2 * Pad
                # Input H coord = Output H coord * Stride H + kh - PAD_H
                x_h = off_m * STRIDE_H + kh - PAD_H
                x_w = off_n * STRIDE_W + kw - PAD_W

                # Mask for valid input spatial coordinates
                mask_h = x_h >= 0 & x_h < H
                mask_w = x_w >= 0 & x_w < W
                mask_spatial = mask_h[:, None] & mask_w[None, :]

                # Load Input X
                # X shape: (B, IC, H, W)
                # Strides: X_STRIDE_B, X_STRIDE_IC, X_STRIDE_H, X_STRIDE_W
                x_offsets = pid_b * X_STRIDE_B + off_ic[None, :] * X_STRIDE_IC + x_h[:, None] * X_STRIDE_H + x_w[None, :] * X_STRIDE_W
                
                # Mask combining channel and spatial validity
                load_mask = mask_ic[None, :] & mask_spatial[:, :]
                
                x_val = tl.load(X + x_offsets, mask=load_mask, other=0.0)

                # Load Weight W
                # W offset for specific kh, kw
                w_offset_base = w_offsets + kh * W_STRIDE_KH + kw * W_STRIDE_KW
                w_val = tl.load(W + w_offset_base, mask=mask_ic[None, :], other=0.0)
                
                # Expand w_val to (1, BLOCK_KC) to match x_val (BLOCK_M*BLOCK_N, BLOCK_KC)
                # tl.dot expects (M, K) and (K, N)
                # Here x_val is (BLOCK_M*BLOCK_N, BLOCK_KC)
                # w_val is (BLOCK_KC,) -> reshape to (BLOCK_KC, 1)
                
                # Perform Dot Product
                # x_val: (BLOCK_M*BLOCK_N, BLOCK_KC)
                # w_val: (BLOCK_KC, 1)
                # result: (BLOCK_M*BLOCK_N, 1)
                
                # Triton tl.dot requires matrices. 
                # We can use tl.sum(x_val * w_val[None, :], axis=1)
                
                partial_sum = tl.sum(x_val * w_val[None, :], axis=1)
                
                # Accumulate
                acc += partial_sum

    # Add Bias
    if tl.numel(Bias) > 0:
        bias_val = tl.load(Bias + pid_oc)
        acc += bias_val

    # Store Output
    out_offsets = pid_b * OUT_STRIDE_B + pid_oc * OUT_STRIDE_OC + off_m[:, None] * OUT_STRIDE_OH + off_n[None, :] * OUT_STRIDE_OW
    tl.store(Out + out_offsets, acc, mask=mask_2d)

def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                  stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1)):
    """
    Wrapper for the Triton 2D Convolution kernel.
    """
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, IC, H, W = x.shape
    OC, IC_w, KH, KW = weight.shape
    
    assert IC == IC_w
    assert dilation == (1, 1), "Dilation != 1 is not supported in this custom kernel"
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    # Calculate output dimensions
    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1
    
    out = torch.empty((B, OC, OH, OW), device=x.device, dtype=torch.float32)
    
    # Block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_KC = 16 # Input channels per block
    
    # Grid configuration
    # We parallelize over Batch, Output Channels, Output Height, Output Width
    grid_b = B
    grid_oc = OC
    grid_oh = triton.cdiv(OH, BLOCK_M)
    grid_ow = triton.cdiv(OW, BLOCK_N)
    
    grid = (grid_b, grid_oc, grid_oh, grid_ow)
    
    # Strides
    X_STRIDE_B = x.stride(0)
    X_STRIDE_IC = x.stride(1)
    X_STRIDE_H = x.stride(2)
    X_STRIDE_W = x.stride(3)
    
    W_STRIDE_OC = weight.stride(0)
    W_STRIDE_IC = weight.stride(1)
    W_STRIDE_KH = weight.stride(2)
    W_STRIDE_KW = weight.stride(3)
    
    OUT_STRIDE_B = out.stride(0)
    OUT_STRIDE_OC = out.stride(1)
    OUT_STRIDE_OH = out.stride(2)
    OUT_STRIDE_OW = out.stride(3)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, out,
        X_STRIDE_B, X_STRIDE_IC, X_STRIDE_H, X_STRIDE_W,
        W_STRIDE_OC, W_STRIDE_IC, W_STRIDE_KH, W_STRIDE_KW,
        OUT_STRIDE_B, OUT_STRIDE_OC, OUT_STRIDE_OH, OUT_STRIDE_OW,
        B, IC, H, W, OC, KH, KW,
        OH, OW,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK_M, BLOCK_N, BLOCK_KC
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # We keep the PyTorch Conv2d to easily extract weights and biases, 
        # but we will override the forward pass to use our Triton kernel.
        # Note: The provided architecture uses groups=1. This kernel assumes groups=1.
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        # Ensure parameters are contiguous for Triton
        self.conv2d.weight.data = self.conv2d.weight.data.contiguous()
        if self.conv2d.bias is not None:
            self.conv2d.bias.data = self.conv2d.bias.data.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom Triton implementation
        return triton_conv2d(
            x=x,
            weight=self.conv2d.weight,
            bias=self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation
        )