import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,     # Input tensor (B, C, H, W)
    weight_ptr,    # Depthwise weight tensor (C, 1, KH, KW)
    output_ptr,    # Output tensor (B, C, H_out, W_out)
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate output dimensions
    output_size = out_height * out_width
    
    # Each block processes CHANNELS_PER_BLOCK channels
    channel_start = channel_idx * CHANNELS_PER_BLOCK
    
    # Shared memory for input tile
    tile_size = kernel_height * kernel_width
    input_tile = tl.shared_memory(dtype=tl.float32, shape=(tile_size,))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process output elements in blocks
    for out_idx in range(0, output_size, BLOCK_SIZE):
        # Calculate output position
        out_pos = out_idx + tl.arange(0, BLOCK_SIZE)
        mask = out_pos < output_size
        
        # Convert linear output index to (h, w)
        out_h = out_pos // out_width
        out_w = out_pos % out_width
        
        # Calculate input positions
        in_h_start = out_h * stride_h - padding_h
        in_w_start = out_w * stride_w - padding_w
        
        # Initialize accumulator for this block
        block_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Process kernel elements
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                ih = in_h_start + kh
                iw = in_w_start + kw
                
                # Check bounds
                in_bounds = (ih >= 0) & (ih < in_height) & (iw >= 0) & (iw < in_width)
                
                # Load input value
                input_idx = batch_idx * (in_channels * in_height * in_width) + \
                           channel_start * (in_height * in_width) + \
                           ih * in_width + iw
                
                # Load weight value
                weight_idx = channel_start * (kernel_height * kernel_width) + \
                            kh * kernel_width + kw
                
                # Load input and weight
                input_val = tl.load(input_ptr + input_idx, mask=in_bounds & mask, other=0.0)
                weight_val = tl.load(weight_ptr + weight_idx)
                
                # Accumulate
                block_acc += input_val * weight_val
        
        # Store result
        output_idx = batch_idx * (in_channels * out_height * out_width) + \
                    channel_start * (out_height * out_width) + out_pos
        tl.store(output_ptr + output_idx, block_acc, mask=mask)

@triton.jit
def pointwise_conv2d_kernel(
    input_ptr,     # Input tensor (B, C, H, W)
    weight_ptr,    # Pointwise weight tensor (out_channels, in_channels, 1, 1)
    output_ptr,    # Output tensor (B, out_channels, H, W)
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    
    # Flatten spatial dimensions
    total_elements = height * width
    
    # Process elements in blocks
    for i in range(0, total_elements, BLOCK_SIZE):
        # Calculate output position
        pos = i + tl.arange(0, BLOCK_SIZE)
        mask = pos < total_elements
        
        # Calculate output index
        output_idx = batch_idx * (out_channels * height * width) + \
                    out_channel_idx * (height * width) + pos
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Accumulate over input channels
        for c in range(in_channels):
            # Calculate input index
            input_idx = batch_idx * (in_channels * height * width) + \
                       c * (height * width) + pos
            
            # Calculate weight index
            weight_idx = out_channel_idx * in_channels + c
            
            # Load values and accumulate
            input_val = tl.load(input_ptr + input_idx, mask=mask)
            weight_val = tl.load(weight_ptr + weight_idx)
            acc += input_val * weight_val
        
        # Store result
        tl.store(output_ptr + output_idx, acc, mask=mask)

def triton_depthwise_conv2d(input_tensor, weight, stride=1, padding=0, dilation=1):
    """Triton kernel for depthwise convolution"""
    assert input_tensor.dim() == 4, "Input tensor must be 4D (B, C, H, W)"
    assert weight.dim() == 4, "Weight tensor must be 4D (C, 1, KH, KW)"
    
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - (kernel_height - 1) * dilation - 1) // stride + 1
    out_width = (in_width + 2 * padding - (kernel_width - 1) * dilation - 1) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, out_height, out_width, dtype=torch.float32, device=input_tensor.device)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define block sizes
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 1
    
    # Grid dimensions
    grid = (
        batch_size,
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_height,
        kernel_width,
        stride,
        stride,
        padding,
        padding,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    
    return output

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """Triton kernel for pointwise convolution (1x1)"""
    assert input_tensor.dim() == 4, "Input tensor must be 4D (B, C, H, W)"
    assert weight.dim() == 4, "Weight tensor must be 4D (out_channels, in_channels, 1, 1)"
    
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, _, _ = weight.shape
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height, width, dtype=torch.float32, device=input_tensor.device)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define block sizes
    BLOCK_SIZE = 256
    
    # Grid dimensions
    grid = (
        batch_size,
        out_channels
    )
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias_param', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Apply depthwise convolution with Triton kernel
        x = triton_depthwise_conv2d(x, self.depthwise_weight, self.stride, self.padding, self.dilation)
        
        # Apply pointwise convolution with Triton kernel
        x = triton_pointwise_conv2d(x, self.pointwise_weight, self.bias_param)
        
        return x