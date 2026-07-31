import torch
import triton
import triton.language as tl

@triton.jit
def avg_pool_1d_kernel(
    x_ptr,
    out_ptr,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # pid_channel iterates over batch * channels
    pid_channel = tl.program_id(0)
    pid_spatial = tl.program_id(1)

    # Calculate the global spatial offsets handled by this program
    offsets = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_length

    # Base pointer for the current (batch, channel) in input and output
    # Input is contiguous in memory: shape (B, C, L_in)
    # So offset for channel k is k * input_length
    base_in_ptr = x_ptr + pid_channel * input_length
    base_out_ptr = out_ptr + pid_channel * output_length

    # Accumulator for the sum
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over the kernel window
    for k in range(kernel_size):
        # Calculate input indices corresponding to the current kernel offset k
        # input_idx = output_offset * stride - padding + k
        input_idx = offsets * stride - padding + k
        
        # Mask for valid input indices
        in_mask = mask & (input_idx >= 0) & (input_idx < input_length)
        
        # Load input values, using 0.0 for out-of-bound indices
        val = tl.load(base_in_ptr + input_idx, mask=in_mask, other=0.0)
        acc += val

    # Compute average
    avg = acc / kernel_size

    # Store to output
    tl.store(base_out_ptr + offsets, avg, mask=mask)

def triton_avg_pool_1d(x, kernel_size, stride, padding):
    batch_size, in_channels, input_length = x.shape
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    x = x.contiguous()
    out = torch.empty((batch_size, in_channels, output_length), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 256
    grid = (batch_size * in_channels, triton.cdiv(output_length, BLOCK_SIZE))
    
    avg_pool_1d_kernel[grid](
        x, out,
        input_length, output_length,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool_1d(x, self.kernel_size, self.stride, self.padding)