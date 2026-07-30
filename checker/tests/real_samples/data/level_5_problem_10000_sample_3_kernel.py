import torch
import torch.nn as nn
from torch.nn import Parameter
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weight (alpha)
    B,  # pointer to the bias (beta)
    M,  # pointer to the mean
    V,  # pointer to the variance
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # The row is the one blocked by the program id.
    row = tl.program_id(0)
    # Create a mask to handle edge cases if N is not divisible by BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load input data for this row
    x = tl.load(X + row * N + cols, mask=mask, other=0.)
    
    # Compute mean
    sum_x = tl.sum(x * mask, axis=0)
    mean = sum_x / N
    
    # Compute variance
    x_shifted = x - mean
    var = tl.sum(x_shifted * x_shifted * mask, axis=0) / N
    
    # Compute output
    std = tl.sqrt(var + eps)
    y = (x - mean) / std * tl.load(W + cols, mask=mask, other=0.) + tl.load(B + cols, mask=mask, other=0.)
    
    # Store output
    tl.store(Y + row * N + cols, y, mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    """
    Triton implementation of LayerNorm.
    
    Args:
        x: input tensor of shape (batch_size, seq_len, hidden_size)
        weight: scale parameter of shape (hidden_size,)
        bias: bias parameter of shape (hidden_size,)
        eps: epsilon for numerical stability
        
    Returns:
        normalized tensor of same shape as x
    """
    # Reshape to 2D: (batch_size * seq_len, hidden_size)
    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    
    batch_size, hidden_size = x.shape
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Ensure hidden_size is a power of 2 for optimal performance, or use dynamic blocks
    # For simplicity, we'll use a block size that works well in practice
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid = (batch_size,)
    
    # Launch kernel
    layer_norm_kernel[grid](
        x, output, weight, bias,
        None, None,  # We don't need to store mean/var in this version
        hidden_size, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape back to original dimensions
    return output.view(*original_shape)


class LayerNormTriton(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormTriton, self).__init__()
        self.alpha = Parameter(torch.ones(hidden_size))
        self.beta = Parameter(torch.zeros(hidden_size))
        self.eps = eps
        
    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        return triton_layer_norm(x, self.alpha, self.beta, self.eps)


class ModelNew(nn.Module):
    def __init__(self, hidden_size, eps=1e-5) -> None:
        super().__init__()
        self.layer_norm = LayerNormTriton(hidden_size, eps)
    
    def forward(self, x):
        return self.layer_norm(x)