import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done by translating the 2D program ID `pid` to a 1D ID.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    a_block_start = pid_m * BLOCK_SIZE_M
    b_block_start = pid_n * BLOCK_SIZE_N
    a_ptrs = a_ptr + (a_block_start * stride_am + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_am) + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak
    b_ptrs = b_ptr + (b_block_start * stride_bn + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk) + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(a_ptrs, mask=k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=k + tl.arange(0, BLOCK_SIZE_K)[:, None] < K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply activation function if specified
    if ACTIVATION == "relu":
        accumulator = tl.where(accumulator > 0, accumulator, 0.0)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with the accumulator.
    c_block_start = pid_m * BLOCK_SIZE_M
    c_ptrs = c_ptr + (c_block_start * stride_cm + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_cm) + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_cn
    c_mask = (c_block_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a, b):
    """
    Performs matrix multiplication of two tensors using Triton kernel.
    
    Args:
        a (torch.Tensor): Input tensor of shape (M, K)
        b (torch.Tensor): Input tensor of shape (K, N)
        
    Returns:
        torch.Tensor: Output tensor of shape (M, N)
    """
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    # Explicitly set device to ensure tensors are on the same device
    device = a.device
    b = b.to(device)
    
    # Create output tensor
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=device, dtype=torch.float32)
    
    # Define parameters for the kernel
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()
    
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    
    # Launch the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=None,
    )
    return c

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using Triton kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        # Reshape A to 2D for matmul
        A_reshaped = A.view(-1, A.size(-1))  # Shape: (N*M, K)
        # Perform matrix multiplication
        result_2d = triton_matmul(A_reshaped, B)  # Shape: (N*M, L)
        # Reshape back to 3D
        result = result_2d.view(A.shape[0], A.shape[1], B.shape[1])  # Shape: (N, M, L)
        return result