import torch
import triton
import triton.language as tl

@triton.jit
def upper_tri_matmul_kernel(
    A, B, C,
    N,
    stride_am, stride_ak,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows of C and cols of A
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Offsets for cols of C and rows of B
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K
    for pid_k in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Mask for A: i <= k  (upper triangular)
        # offs_m is [BLOCK_SIZE_M], offs_k is [BLOCK_SIZE_K]
        # We need to broadcast to [BLOCK_SIZE_M, BLOCK_SIZE_K]
        mask_a = (offs_m[:, None] <= offs_k[None, :]) & (offs_k[None, :] < N)
        a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=mask_a, other=0.0)
        
        # Mask for B: k <= j  (upper triangular)
        # offs_k is [BLOCK_SIZE_K], offs_n is [BLOCK_SIZE_N]
        # We need to broadcast to [BLOCK_SIZE_K, BLOCK_SIZE_N]
        mask_b = (offs_k[:, None] <= offs_n[None, :]) & (offs_k[:, None] < N)
        b = tl.load(B + offs_k[:, None] * stride_bm + offs_n[None, :] * stride_bn, mask=mask_b, other=0.0)
        
        # Multiply and accumulate
        # We only accumulate if both A and B elements are valid (non-zero in upper triangular sense)
        # However, since we loaded 0.0 for invalid parts, simple matmul accumulation works:
        # acc += a @ b
        # But wait, tl.dot expects 2D tensors. a is [BM, BK], b is [BK, BN].
        # tl.dot(a, b) will compute the block-wise dot product.
        # Since invalid parts are 0.0, they contribute 0 to the sum. This is correct.
        acc += tl.dot(a, b)
        
        # Advance offs_k is not needed here because offs_k is recalculated in the loop
        # or we can just rely on the loop variable pid_k. 
        # In Triton, we usually just recalculate offsets or increment. 
        # Here we just recalculate via pid_k * BLOCK_SIZE_K... wait, 
        # actually, inside the loop, we just need to ensure offs_k advances.
        # The current implementation recalculates offs_k from scratch each iteration, which is fine.
        # But for performance, usually we do:
        # offs_k = tl.arange(0, BLOCK_SIZE_K) + pid_k * BLOCK_SIZE_K
        # But since pid_k changes, we need to update offs_k.
        # Actually, the range function in Triton loop is slightly different.
        # Let's stick to the standard pattern:
        # offs_k = tl.arange(0, BLOCK_SIZE_K) + pid_k * BLOCK_SIZE_K
        # But we need to update pid_k? No, the loop handles pid_k.
        # We need to update offs_k to reflect the current block.
        # The code above: offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        # This is correct for each iteration.

    # Mask for C: i <= j (upper triangular)
    # offs_m is [BM], offs_n is [BN]
    mask_c = (offs_m[:, None] <= offs_n[None, :]) & (offs_m[:, None] < N) & (offs_n[None, :] < N)
    
    # Store
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=mask_c)


def triton_upper_tri_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication of two upper triangular matrices using Triton.
    Fuses the matmul and triu operations.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Input matrices must have the same shape."
    
    N = A.shape[0]
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    # Calculate grid dimensions
    grid_m = triton.cdiv(N, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    grid = (grid_m, grid_n)
    
    # Launch kernel
    upper_tri_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    Optimized with Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using Triton.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_upper_tri_matmul(A, B)


N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.rand(N, N, device='cuda'))
    B = torch.triu(torch.rand(N, N, device='cuda'))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []