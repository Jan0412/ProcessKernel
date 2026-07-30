import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softplus_tanh_mul_bn_kernel(
    input_ptr,  # Input tensor [N, C, H, W]
    weight_ptr,  # Weight tensor [C_out, C_in, K, K]
    bias_ptr,  # Bias tensor [C_out]
    running_mean_ptr,  # Running mean [C_out]
    running_var_ptr,  # Running var [C_out]
    gamma_ptr,  # Gamma [C_out]
    beta_ptr,  # Beta [C_out]
    output_ptr,  # Output tensor [N, C_out, H, W]
    stride_input_n, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_c_out, stride_weight_c_in, stride_weight_k, stride_weight_k,
    stride_output_n, stride_output_c, stride_output_h, stride_output_w,
    n, c_in, h_in, w_in, c_out, h_out, w_out, k,
    eps,
    BLOCK_N_SIZE: tl.constexpr, BLOCK_C_SIZE: tl.constexpr,
    BLOCK_H_SIZE: tl.constexpr, BLOCK_W_SIZE: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute output coordinates
    n_offsets = pid_n * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
    c_offsets = pid_c * BLOCK_C_SIZE + tl.arange(0, BLOCK_C_SIZE)
    h_offsets = pid_h * BLOCK_H_SIZE + tl.arange(0, BLOCK_H_SIZE)
    w_offsets = pid_w * BLOCK_W_SIZE + tl.arange(0, BLOCK_W_SIZE)

    # Create masks for valid coordinates
    n_mask = n_offsets < n
    c_mask = c_offsets < c_out
    h_mask = h_offsets < h_out
    w_mask = w_offsets < w_out

    # Initialize accumulator
    acc = tl.zeros((BLOCK_N_SIZE, BLOCK_C_SIZE, BLOCK_H_SIZE, BLOCK_W_SIZE), dtype=tl.float32)

    # Iterate over input channels
    for c_in_idx in range(0, c_in, BLOCK_C_SIZE):
        c_in_offsets = c_in_idx + tl.arange(0, BLOCK_C_SIZE)
        c_in_mask = c_in_offsets < c_in

        # Load input tile
        input_ptrs = input_ptr + (
            n_offsets[:, None, None, None] * stride_input_n +
            c_in_offsets[None, :, None, None] * stride_input_c +
            h_offsets[None, None, :, None] * stride_input_h +
            w_offsets[None, None, None, :] * stride_input_w
        )
        input_vals = tl.load(input_ptrs, mask=(n_mask[:, None, None, None] & c_in_mask[None, :, None, None] & h_mask[None, None, :, None] & w_mask[None, None, None, :]), other=0.0)

        # Load weight tile
        weight_ptrs = weight_ptr + (
            c_offsets[:, None, None, None] * stride_weight_c_out +
            c_in_offsets[None, :, None, None] * stride_weight_c_in +
            tl.arange(0, k)[None, None, :, None] * stride_weight_k +
            tl.arange(0, k)[None, None, None, :] * stride_weight_k
        )
        weight_vals = tl.load(weight_ptrs, mask=(c_mask[:, None, None, None] & c_in_mask[None, :, None, None]), other=0.0)

        # Perform convolution
        acc += tl.sum(input_vals * weight_vals, axis=1)

    # Add bias
    bias_vals = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)
    acc = acc + bias_vals[None, :, None, None]

    # Apply softplus
    acc = tl.log(tl.exp(acc) + 1.0)

    # Apply tanh
    acc = tl.tanh(acc)

    # Multiply by input (element-wise)
    input_ptrs = input_ptr + (
        n_offsets[:, None, None, None] * stride_input_n +
        c_offsets[None, :, None, None] * stride_input_c +
        h_offsets[None, None, :, None] * stride_input_h +
        w_offsets[None, None, None, :] * stride_input_w
    )
    input_vals = tl.load(input_ptrs, mask=(n_mask[:, None, None, None] & c_mask[None, :, None, None] & h_mask[None, None, :, None] & w_mask[None, None, None, :]), other=0.0)
    acc = acc * input_vals

    # Apply BatchNorm
    running_mean_vals = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0.0)
    running_var_vals = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=0.0)
    gamma_vals = tl.load(gamma_ptr + c_offsets, mask=c_mask, other=0.0)
    beta_vals = tl.load(beta_ptr + c_offsets, mask=c_mask, other=0.0)

    acc = (acc - running_mean_vals[None, :, None, None]) / tl.sqrt(running_var_vals[None, :, None, None] + eps) * gamma_vals[None, :, None, None] + beta_vals[None, :, None, None]

    # Store output
    output_ptrs = output_ptr + (
        n_offsets[:, None, None, None] * stride_output_n +
        c_offsets[None, :, None, None] * stride_output_c +
        h_offsets[None, None, :, None] * stride_output_h +
        w_offsets[None, None, None, :] * stride_output_w
    )
    tl.store(output_ptrs, acc, mask=(n_mask[:, None, None, None] & c_mask[None, :, None, None] & h_mask[None, None, :, None] & w_mask[None, None, None, :]))


def fused_conv_softplus_tanh_mul_bn(input, weight, bias, running_mean, running_var, gamma, beta, eps):
    # Ensure inputs are contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Get tensor shapes
    n, c_in, h_in, w_in = input.shape
    c_out, k, _, _ = weight.shape
    h_out = h_in - k + 1
    w_out = w_in - k + 1

    # Create output tensor
    output = torch.empty(n, c_out, h_out, w_out, device=input.device, dtype=input.dtype)

    # Block sizes
    BLOCK_N_SIZE = 1
    BLOCK_C_SIZE = 32
    BLOCK_H_SIZE = 16
    BLOCK_W_SIZE = 16

    # Grid
    grid = (
        n,
        triton.cdiv(c_out, BLOCK_C_SIZE),
        triton.cdiv(h_out, BLOCK_H_SIZE),
        triton.cdiv(w_out, BLOCK_W_SIZE),
    )

    # Launch kernel
    fused_conv_softplus_tanh_mul_bn_kernel[grid](
        input, weight, bias, running_mean, running_var, gamma, beta, output,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        n, c_in, h_in, w_in, c_out, h_out, w_out, k,
        eps,
        BLOCK_N_SIZE=BLOCK_N_SIZE, BLOCK_C_SIZE=BLOCK_C_SIZE,
        BLOCK_H_SIZE=BLOCK_H_SIZE, BLOCK_W_SIZE=BLOCK_W_SIZE,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        # Get conv parameters
        weight = self.conv.weight
        bias = self.conv.bias

        # Get bn parameters
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps

        # Call fused kernel
        return fused_conv_softplus_tanh_mul_bn(x, weight, bias, running_mean, running_var, gamma, beta, eps)