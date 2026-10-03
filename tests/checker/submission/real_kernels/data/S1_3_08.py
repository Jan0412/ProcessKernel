import torch
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    X_ptr,
    Y_ptr,
    dim,
    stride_x,
    stride_y,
    BLOCK_DIM: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Pointers to the start of the current row
    row_x_ptr = X_ptr + pid * stride_x
    row_y_ptr = Y_ptr + pid * stride_y
    
    norm = 0.0
    
    # Number of blocks to process
    num_blocks = tl.cdiv(dim, BLOCK_DIM)
    
    # Pass 1: Compute L2 norm
    for i in range(num_blocks):
        offsets = i * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        mask = offsets < dim
        
        # Load as float32. If input is already float32, this is a no-op. 
        # If input is float16, this converts it. This avoids the host-side .to() kernel.
        x = tl.load(row_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        norm += tl.sum(x * x)
        
    # Compute norm
    norm = tl.sqrt(norm)
    
    # Avoid division by zero
    norm = tl.where(norm > 0, norm, 1.0)
    
    # Pass 2: Divide by norm and store
    for i in range(num_blocks):
        offsets = i * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        mask = offsets < dim
        
        # Load again
        x = tl.load(row_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Divide and store
        out = x / norm
        tl.store(row_y_ptr + offsets, out, mask=mask)

def triton_l2_norm(x):
    # x is (batch_size, dim)
    assert x.is_cuda
    x = x.contiguous() # Ensure contiguous layout without casting
    
    batch_size, dim = x.shape
    out = torch.empty_like(x) # Keeps dtype of x
    
    BLOCK_DIM = 1024
    
    grid = (batch_size,)
    
    l2_norm_kernel[grid](
        x,
        out,
        dim,
        x.stride(0),
        out.stride(0),
        BLOCK_DIM=BLOCK_DIM
    )
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)