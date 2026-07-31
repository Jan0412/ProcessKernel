import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr,   # Input tensor (N, C, H, W)
    weight_ptr,  # Weight tensor (out_channels, in_channels, kernel_h, kernel_w)
    output_ptr,  # Output tensor (N, out_channels, out_h, out_w)
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_oc, weight_stride_ic, weight_stride_kh, weight_stride_kw,
    output_stride_n, output_stride_oc, output_stride_h, output_stride_w,
    N, C, H, W, 
    out_channels, out_h, out_w,
    kernel_h, kernel_w,
    padding_h, padding_w,
    stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    # Each program handles one output element
    output_idx = pid
    
    # Calculate output coordinates
    out_w_idx = output_idx % out_w
    out_h_idx = (output_idx // out_w) % out_h
    out_c_idx = (output_idx // (out_w * out_h)) % out_channels
    batch_idx = output_idx // (out_w * out_h * out_channels)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(C):
        # Loop over kernel elements
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input coordinates
                ih = out_h_idx * stride_h + kh - padding_h
                iw = out_w_idx * stride_w + kw - padding_w
                
                # Check bounds
                if ih >= 0 and ih < H and iw >= 0 and iw < W:
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_idx * input_stride_n + 
                                       c * input_stride_c + 
                                       ih * input_stride_h + 
                                       iw * input_stride_w)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        out_c_idx * weight_stride_oc + 
                                        c * weight_stride_ic + 
                                        kh * weight_stride_kh + 
                                        kw * weight_stride_kw)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + 
             batch_idx * output_stride_n + 
             out_c_idx * output_stride_oc + 
             out_h_idx * output_stride_h + 
             out_w_idx * output_stride_w, 
             acc[0])

def triton_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton-based Conv2D implementation
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda, "Weight tensor must be on CUDA"
    
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
    output = torch.empty(N, out_channels, out_h, out_w, dtype=torch.float32, device='cuda')
    
    # Calculate strides
    input_stride_n, input_stride_c, input_stride_h, input_stride_w = input_tensor.stride()
    weight_stride_oc, weight_stride_ic, weight_stride_kh, weight_stride_kw = weight.stride()
    output_stride_n, output_stride_oc, output_stride_h, output_stride_w = output.stride()
    
    # Launch kernel
    grid_size = N * out_channels * out_h * out_w
    BLOCK_SIZE = 128
    
    # Create grid function
    grid = lambda meta: (grid_size,)
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        weight_stride_oc, weight_stride_ic, weight_stride_kh, weight_stride_kw,
        output_stride_n, output_stride_oc, output_stride_h, output_stride_w,
        N, C, H, W,
        out_channels, out_h, out_w,
        kernel_h, kernel_w,
        padding, padding,
        stride, stride,
        BLOCK_SIZE
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
        nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        # Replace the standard conv2d with our Triton implementation
        return triton_conv2d(x, self.conv1.weight, self.conv1.bias, 
                           stride=self.conv1.stride, padding=self.conv1.padding)