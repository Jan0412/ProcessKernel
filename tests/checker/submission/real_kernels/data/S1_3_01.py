import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr,   # Input tensor [N, C, H, W]
    weight_ptr,  # Weight tensor [out_channels, in_channels, kernel_h, kernel_w]
    output_ptr,  # Output tensor [N, out_channels, out_h, out_w]
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_oc, weight_stride_ic, weight_stride_kh, weight_stride_kw,
    output_stride_n, output_stride_oc, output_stride_h, output_stride_w,
    N, C, H, W, out_channels, out_h, out_w, kernel_h, kernel_w,
    padding_h, padding_w,
    stride_h, stride_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get block indices
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (input channels)
    for k in range(0, C, BLOCK_SIZE_K):
        # Load input tile [BLOCK_SIZE_M, BLOCK_SIZE_K]
        input_offset = pid_n * input_stride_n + k * input_stride_c + \
                       pid_h * stride_h * input_stride_h + pid_w * stride_w * input_stride_w
        input_tile = tl.load(input_ptr + input_offset, mask=(k + tl.arange(0, BLOCK_SIZE_K)) < C)
        
        # Load weight tile [BLOCK_SIZE_K, BLOCK_SIZE_N]
        weight_offset = pid_oc * weight_stride_oc + k * weight_stride_ic
        weight_tile = tl.load(weight_ptr + weight_offset, mask=(k + tl.arange(0, BLOCK_SIZE_K)) < C)
        
        # Perform matrix multiplication
        acc += tl.dot(input_tile, weight_tile)
    
    # Store result
    output_offset = pid_n * output_stride_n + pid_oc * output_stride_oc + \
                    pid_h * output_stride_h + pid_w * output_stride_w
    tl.store(output_ptr + output_offset, acc)

def triton_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton-based Conv2d implementation
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    N, C, H, W = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1
    
    # Create output tensor
    output = torch.empty(N, out_channels, out_h, out_w, dtype=torch.float32, device=input_tensor.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Create grid
    grid = (
        N,
        out_channels,
        out_h,
        out_w
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N, C, H, W, out_channels, out_h, out_w, kernel_h, kernel_w,
        padding, padding,
        stride, stride,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # Initialize weights to match PyTorch's default initialization
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        # Replace the standard conv2d with our Triton implementation
        return triton_conv2d(x, self.conv1.weight, self.conv1.bias, 
                           stride=self.conv1.stride, padding=self.conv1.padding)