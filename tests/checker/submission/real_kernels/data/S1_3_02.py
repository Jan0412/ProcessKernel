import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr,    # Input tensor pointer
    weight_ptr,   # Weight tensor pointer
    output_ptr,   # Output tensor pointer
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Calculate output coordinates
    output_row = pid_m * BLOCK_SIZE_M
    output_col = pid_n * BLOCK_SIZE_N
    channel_idx = pid_k * BLOCK_SIZE_K
    
    # Create output tile
    output_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over kernel spatial dimensions
    for kh in range(0, kernel_height):
        for kw in range(0, kernel_width):
            # Calculate input coordinates with padding
            input_row = output_row * stride_h + kh - padding_h
            input_col = output_col * stride_w + kw - padding_w
            
            # Check bounds for input
            input_valid = (input_row >= 0) & (input_row < input_height) & \
                         (input_col >= 0) & (input_col < input_width)
            
            # Load input data
            input_data = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            if input_valid:
                input_data = tl.load(input_ptr + 
                                   input_row * input_width + input_col, 
                                   mask=input_valid, other=0.0)
            
            # Load weight data
            weight_data = tl.load(weight_ptr + 
                                channel_idx * kernel_height * kernel_width + 
                                kh * kernel_width + kw, 
                                mask=(channel_idx < in_channels), other=0.0)
            
            # Compute convolution
            output_tile += input_data * weight_data
    
    # Store output
    tl.store(output_ptr + 
             output_row * output_width + output_col, 
             output_tile, 
             mask=(output_row < output_height) & (output_col < output_width))

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0)):
    """
    Triton-based Conv2D implementation
    """
    # Ensure inputs are on GPU and contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - kernel_height) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - kernel_width) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, 
                        dtype=torch.float32, device=input_tensor.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions
    grid_m = (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (in_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    grid = (grid_m, grid_n, grid_k)
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # Initialize weights to match PyTorch's default initialization
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = triton_conv2d(x, self.conv1.weight, self.conv1.bias, 
                         stride=self.conv1.stride, padding=self.conv1.padding)
        return x