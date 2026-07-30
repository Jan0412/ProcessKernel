import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_forward_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_stride,
    input_stride,
    weight_stride,
    bias_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Base pointers for this row
    input_row_ptr = input_ptr + row_idx * input_stride
    output_row_ptr = output_ptr + row_idx * output_stride
    weight_ptr = weight_ptr + row_idx * weight_stride # if weights are per-row, otherwise stride 0
    bias_ptr = bias_ptr + row_idx * bias_stride
    
    # Mean and Variance Calculation
    mean = 0.0
    var = 0.0
    
    # Loop over columns in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load inputs
        input_vals = tl.load(input_row_ptr + col_offsets, mask=mask, other=0.0)
        
        # Accumulate mean
        mean += tl.sum(input_vals) / n_cols
        
    # Second pass for variance
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        input_vals = tl.load(input_row_ptr + col_offsets, mask=mask, other=0.0)
        diff = input_vals - mean
        var += tl.sum(diff * diff) / n_cols
        
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Normalization and apply weight/bias
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        input_vals = tl.load(input_row_ptr + col_offsets, mask=mask, other=0.0)
        
        # Load weight and bias
        weight = tl.load(weight_ptr)
        bias = tl.load(bias_ptr)
        
        # Normalize
        normalized = (input_vals - mean) * inv_std
        output_vals = normalized * weight + bias
        
        tl.store(output_row_ptr + col_offsets, output_vals, mask=mask)

def triton_layer_norm(x, weight, bias, eps):
    B, F, D1, D2 = x.shape
    n_rows = B * F
    n_cols = D1 * D2
    
    output = torch.empty_like(x)
    
    grid = lambda META: (n_rows,)
    
    layer_norm_forward_kernel[grid](
        output,
        x,
        weight,
        bias,
        output.stride(0),
        x.stride(0),
        weight.stride(0),
        bias.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=128,
    )
    return output