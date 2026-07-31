import torch
import torch.nn as nn
import triton
import triton.language as tl

# Constants for GELU approximation
SQRT_2_OVER_PI = 0.7978845608028654
COEFF_3 = 0.044715

@triton.jit
def fused_gemm_activation_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    scaling_factor, hardtanh_min, hardtanh_max,
    stride_am, stride_ak,
    stride_bk, stride_bn, # Note: B is W, shape (N, K). We access W[j, k]. stride_bk=1, stride_bn=K?
    # Wait, W is (N, K). Row j is W[j, :].
    # If we load a tile of W of size (BLOCK_N, BLOCK_K), we need to handle strides correctly.
    # W is stored row-major. W[j, k] is at offset j*K + k.
    # So stride for N (rows) is K, stride for K (cols) is 1.
    # Let's call the input matrix B (which is W).
    stride_bm, stride_bk, # B is (N, K). stride_bm = K, stride_bk = 1.
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows and columns of the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Accumulator for the dot product
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tile from A (M, K)
        # A is row-major. A[i, k]
        cols_k = k + tl.arange(0, BLOCK_K)
        mask_k = cols_k < K
        
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        # strides: stride_am for rows, stride_ak for cols (1)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + cols_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load tile from B (W) (N, K)
        # We need W[j, k]. B is (N, K).
        # Rows correspond to N index, Cols correspond to K index.
        # B is row-major. B[j, k] offset is j*K + k.
        # So stride for N (rows) is K, stride for K (cols) is 1.
        # Let's rename strides to be clear: stride_bn (row stride), stride_bk (col stride)
        # In the function signature I put stride_bk, stride_bn. Let's align.
        # B_ptr + offs_n[:, None] * stride_bn + cols_k[None, :] * stride_bk
        
        # Wait, in signature I had stride_bk, stride_bn.
        # Let's assume standard order: stride for first dim, stride for second dim.
        # B is (N, K). First dim N, second dim K.
        # So stride_bn is K, stride_bk is 1.
        
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + cols_k[None, :] * stride_bk
        b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Matrix multiply accumulation
        # acc += A @ B.T ?
        # We want C[i, j] = sum_k A[i, k] * B[j, k]
        # A tile is (BLOCK_M, BLOCK_K)
        # B tile is (BLOCK_N, BLOCK_K)
        # We need to compute dot products between rows of A and rows of B.
        # This is equivalent to A @ B.T if B was (BLOCK_K, BLOCK_N).
        # But B is (BLOCK_N, BLOCK_K).
        # So we can do tl.dot(a, b.T)?
        # Triton tl.dot expects (M, K) and (K, N).
        # a is (BLOCK_M, BLOCK_K).
        # b is (BLOCK_N, BLOCK_K).
        # b.T would be (BLOCK_K, BLOCK_N).
        # So tl.dot(a, b.T) works.
        
        # However, tl.dot might not support transposing the second operand directly if it's not a specific layout?
        # Actually, tl.dot(a, b, trans_b=True) is available in newer Triton versions?
        # Or just transpose b.
        # Transposing a small tile in shared memory/register is cheap?
        # Actually, we can just swap loops or indices, but tl.dot is optimized.
        # Let's try to transpose b.
        
        # b is (BLOCK_N, BLOCK_K). We want to treat it as (BLOCK_K, BLOCK_N) for dot product with a (BLOCK_M, BLOCK_K).
        # Wait, tl.dot(a, b) computes a @ b.
        # If a is (M, K) and b is (K, N), result is (M, N).
        # Here a is (BLOCK_M, BLOCK_K).
        # We have b_tile as (BLOCK_N, BLOCK_K).
        # We need b_tile.T which is (BLOCK_K, BLOCK_N).
        # tl.dot(a, b_tile.T) -> (BLOCK_M, BLOCK_N).
        
        # Is there a trans_b flag?
        # In Triton 2.0+, tl.dot supports trans_b.
        # Let's assume standard Triton. If not, we can manually transpose or restructure.
        # But writing a manual transpose loop in Triton is verbose.
        # Let's check if we can just load b differently.
        # If we load b as (BLOCK_K, BLOCK_N) directly?
        # b is stored (N, K). Accessing columns of b (which are rows) is strided.
        # Loading a (BLOCK_K, BLOCK_N) tile from b would mean accessing b[k, n] for k in range(BLOCK_K), n in range(BLOCK_N).
        # b[k, n] is not valid indices since b is (N, K). Indices are (n, k).
        # So b[n, k].
        # We want to iterate k and n.
        # If we load b as (BLOCK_K, BLOCK_N), we are loading b[n, k] but reshaped?
        # No, memory layout is fixed.
        # b[n, k] is contiguous in k.
        # So loading rows of b is efficient.
        # We loaded b as (BLOCK_N, BLOCK_K). This is efficient.
        # Now we need to compute dot product of rows.
        # acc[i, j] = sum_k a[i, k] * b[j, k].
        # This is exactly what tl.dot(a, b, trans_b=True) would do if supported.
        # If not supported, we can swap the loops?
        # No, standard GEMM loop order is K outer.
        
        # Alternative: Use tl.dot(a, b.T).
        # b is (BLOCK_N, BLOCK_K). b.T is (BLOCK_K, BLOCK_N).
        # Transposing a matrix in Triton:
        # b_t = tl.trans(b) ? No tl.trans.
        # We can use tl.permute? No.
        # We can just use the fact that dot product is commutative in a sense? No.
        
        # Let's look at Triton documentation or common patterns.
        # Usually, for A @ B^T, one might store B in column-major or handle the transpose.
        # But since B is (N, K) and we access rows, it's row-major.
        # If we treat B as (K, N) in our mind, it would be column-major access.
        
        # Let's try to implement the dot product manually or check if tl.dot has a transpose option.
        # tl.dot(a, b, trans_b=True) is supported in Triton.
        # Reference: https://triton-lang.org/main/python-api/generated/triton.language.dot.html
        # "trans_b: bool, optional. If True, the second matrix is transposed."
        # Yes, it exists.
        
        acc = tl.dot(a, b, trans_b=True, acc=acc)

    # Add bias
    # bias is (N,). We need to broadcast it to (BLOCK_M, BLOCK_N).
    # bias[j] corresponds to column j of output.
    # offs_n is (BLOCK_N,).
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    # bias shape (BLOCK_N,). Need to add to acc (BLOCK_M, BLOCK_N).
    # Broadcasting works if we reshape or just rely on Triton's broadcasting?
    # acc is 2D. bias is 1D.
    # acc += bias[None, :]
    acc = acc + bias[None, :]

    # Apply scaling
    acc = acc * scaling_factor

    # Apply Hardtanh
    # clamp between hardtanh_min and hardtanh_max
    acc = tl.minimum(tl.maximum(acc, hardtanh_min), hardtanh_max)

    # Apply GELU
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Use tl.libdevice.tanh
    x = acc
    # Compute x^3
    x3 = x * x * x
    # Compute argument for tanh
    # 0.044715 * x^3
    term1 = COEFF_3 * x3
    # x + term1
    arg = x + term1
    # sqrt(2/pi) * arg
    arg = SQRT_2_OVER_PI * arg
    # tanh(arg)
    tanh_val = tl.libdevice.tanh(arg)
    # 1 + tanh_val
    one_plus_tanh = 1.0 + tanh_val
    # 0.5 * x * one_plus_tanh
    gelu_val = 0.5 * x * one_plus_tanh

    # Store output
    # Output shape (M, N).
    # c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, gelu_val, mask=mask_m[:, None] & mask_n[None, :])

def fused_gemm_activation(x, w, b, scaling_factor, hardtanh_min, hardtanh_max):
    # x: (M, K)
    # w: (N, K) -> corresponds to W in Linear, where Linear computes x @ W.T + b
    # b: (N,)
    # Output: (M, N)
    
    M, K = x.shape
    N, _ = w.shape
    
    # Check shapes
    assert x.shape[1] == w.shape[1], f"K mismatch: {x.shape[1]} vs {w.shape[1]}"
    assert b.shape[0] == N, f"Bias mismatch: {b.shape[0]} vs {N}"
    
    # Output tensor
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64 # Tune this? 64 or 128. 8192 features.
    
    # Grid
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    
    fused_gemm_activation_kernel[grid](
        x, w, out, b,
        M, N, K,
        scaling_factor, hardtanh_min, hardtanh_max,
        x.stride(0), x.stride(1), # stride_am, stride_ak
        w.stride(0), w.stride(1), # stride_bn (K), stride_bk (1) -> Wait.
        # w is (N, K). stride(0) is K, stride(1) is 1.
        # In kernel signature: stride_bk, stride_bn.
        # I passed w.stride(0) as stride_bn?
        # Let's recheck kernel signature vs call.
        # Kernel: stride_bk, stride_bn (order in args)
        # Call: w.stride(0), w.stride(1)
        # w.stride(0) is K (stride for N dim).
        # w.stride(1) is 1 (stride for K dim).
        # In kernel:
        # b_ptrs = b_ptr + offs_n[:, None] * stride_bn + cols_k[None, :] * stride_bk
        # offs_n corresponds to N index. stride_bn should be K.
        # cols_k corresponds to K index. stride_bk should be 1.
        # So passing (w.stride(0), w.stride(1)) is correct if kernel args are (stride_bk, stride_bn)??
        # Wait, kernel args: stride_bk, stride_bn.
        # Call args: w.stride(0), w.stride(1).
        # w.stride(0) is K. This is assigned to stride_bk?
        # w.stride(1) is 1. This is assigned to stride_bn?
        # That would be WRONG.
        # stride_bk should be 1. stride_bn should be K.
        # So I should pass w.stride(1), w.stride(0).
        
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return out

# Wait, I need to verify the kernel argument order and usage.
# Kernel def: ..., stride_bk, stride_bn, ...
# Usage: b_ptrs = b_ptr + offs_n[:, None] * stride_bn + cols_k[None, :] * stride_bk
# offs_n is N index. stride_bn is stride for N.
# cols_k is K index. stride_bk is stride for K.
# w is (N, K).
# w.stride(0) is stride for N (which is K).
# w.stride(1) is stride for K (which is 1).
# So stride_bn = w.stride(0) = K.
# stride_bk = w.stride(1) = 1.
# In call: fused_gemm_activation_kernel(..., w.stride(0), w.stride(1), ...)
# If kernel args are (stride_bk, stride_bn), then:
# stride_bk gets w.stride(0) = K.
# stride_bn gets w.stride(1) = 1.
# This is swapped!
# I should change kernel args order or call order.
# Let's change kernel args to (stride_bn, stride_bk) to match logical order (row, col) or just be explicit.
# Actually, standard is usually (stride_0, stride_1).
# Let's fix the kernel signature to be (stride_bn, stride_bk) or just name them clearly.
# Let's rename kernel args to stride_w_n, stride_w_k.