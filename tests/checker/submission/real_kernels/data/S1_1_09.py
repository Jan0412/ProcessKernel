import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose_2d_kernel(
    X, W, B, O,
    N, C_in, C_out, H_in, W_in, H_out, W_out, K,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n_cout = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    n = pid_n_cout // C_out
    c_out = pid_n_cout % C_out
    
    hw_idx = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_hw = hw_idx < H_out * W_out
    
    h_out = hw_idx // W_out
    w_out = hw_idx % W_out
    
    h_in_base = h_out * stride - padding
    w_in_base = w_out * stride - padding
    
    h_in_off = h_in_base[:, None] + tl.arange(0, K)[None, :] * dilation
    w_in_off = w_in_base[:, None] + tl.arange(0, K)[None, :] * dilation
    
    mask_h = (h_in_off >= 0) & (h_in_off < H_in)
    mask_w = (w_in_off >= 0) & (w_in_off < W_in)
    mask = mask_h & mask_w
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for c_in in range(C_in):
        x_off = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in_off[:, None] * W_in + w_in_off
        x = tl.load(X + x_off, mask=mask, other=0.0)
        
        w_off = c_in * C_out * K * K + c_out * K * K + tl.arange(0, K)[:, None] * K + tl.arange(0, K)
        w = tl.load(W + w_off)
        
        acc += tl.sum(x * w, -2)
        
    if B is not None:
        bias = tl.load(B + c_out)
        acc += bias
        
    o_off = n * C_out * H_out * W_out + c_out * H_out * W_out + hw_idx
    tl.store(O + o_off, acc, mask=mask_hw)

def triton_conv_transpose_2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    N, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = weight.shape
    assert K_h == K_w, "Only square kernels supported"
    K = K_h
    
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1 + 2 * padding # Wait, standard formula:
    # H_out = (H_in - 1) * stride + 2 * padding - dilation * (K - 1) + 1
    # Let's verify: if S=1, P=0, D=1, K=3 -> H_out = H_in - 2. Correct for transpose?
    # No, ConvTranspose2d output size:
    # H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1 + output_padding
    # Assuming output_padding = 0.
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    grid = (N * C_out, triton.cdiv(H_out * W_out, 256))
    
    conv_transpose_2d_kernel[grid](
        x, weight, bias, out,
        N, C_in, C_out, H_in, W_in, H_out, W_out, K,
        stride, padding, dilation,
        BLOCK_SIZE=256
    )
    return out