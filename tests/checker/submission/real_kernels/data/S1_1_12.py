import torch
import triton
import triton.language as tl

@triton.jit
def cumprod_kernel(
    x_ptr,
    out_ptr,
    M,  # batch_size
    N,  # sequence length (input_shape[0])
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    pid = tl.program_id(0)
    if pid >= M:
        return

    # Pointers to row
    row_start_x = x_ptr + pid * stride_xm
    row_start_out = out_ptr + pid * stride_om

    # Offsets for columns
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load row
    x = tl.load(row_start_x + offsets * stride_xn, mask=mask, other=1.0) # other=1.0 for multiplication identity

    # Compute cumulative product
    out = tl.associative_scan(x, tl.mul)

    # Store result
    tl.store(row_start_out + offsets * stride_on, out, mask=mask)

def triton_cumprod(x, dim):
    # Ensure contiguous
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)
    # Triton scan works best with powers of 2, and BLOCK_SIZE >= N
    # Actually, associative_scan in Triton requires BLOCK_SIZE to be a power of 2 and >= N.
    # Let's adjust BLOCK_SIZE to be at least N and a power of 2.
    # But N=32768 is already a power of 2.
    # We'll use BLOCK_SIZE = N if N is power of 2, else next power of 2.
    # Actually, Triton's associative_scan handles sizes < BLOCK_SIZE correctly with masking.
    # It's safer to set BLOCK_SIZE = N if N is power of 2, or just use a large power of 2.
    # Let's compute it dynamically.
    import math
    BLOCK_SIZE = 1 << (N - 1).bit_length() if N > 0 else 1
    # Ensure BLOCK_SIZE >= N
    if BLOCK_SIZE < N:
        BLOCK_SIZE = BLOCK_SIZE * 2

    grid = (M,)
    cumprod_kernel[grid](
        x, out, M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out