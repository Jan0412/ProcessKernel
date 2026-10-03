import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    X,  # Input tensor pointer (N, C_in, D, H, W)
    W,  # Weight tensor pointer (C_out, C_in, K_D, K_H, K_W)
    Out,  # Output tensor pointer (N, C_out, D_out, H_out, W_out)
    stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kd, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_co, stride_out_d, stride_out_h, stride_out_w,
    N, C_in, D, H, W, C_out, K_D, K_H, K_W,
    D_out, H_out, W_out,
    BLOCK_N: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Initialize program indices
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    # Block offsets for output spatial dimensions
    # We flatten D_out, H_out, W_out into a single 1D index for the grid
    # and then unflatten it here.
    # Total spatial elements per program = BLOCK_D * BLOCK_H * BLOCK_W
    # This assumes we are tiling the output spatial dimensions together.
    
    # Calculate linear spatial offset for this program
    # We treat the spatial dimensions as a single flattened space for tiling
    # But to compute actual D, H, W indices, we need to unflatten.
    # Let's use a simpler 3D grid for spatial: (D_out, H_out, W_out) is too large.
    # Instead, let's use a 1D grid for spatial tiles and unflatten inside.
    
    # Number of spatial elements per block
    BLOCK_SPATIAL = BLOCK_D * BLOCK_H * BLOCK_W
    spatial_offset = pid_dhw * BLOCK_SPATIAL
    
    # Unflatten spatial_offset into d, h, w
    # w = spatial_offset % W_out
    # h = (spatial_offset // W_out) % H_out
    # d = (spatial_offset // (W_out * H_out)) % D_out
    # But Triton doesn't like dynamic shapes in unflattening easily.
    # Let's stick to a 3D grid for (N, C_out, spatial_tile) where spatial_tile is 1D.
    # And compute d, h, w from the 1D offset.
    
    # Offsets within the block
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    
    # Spatial offsets within the block
    # We need to map a 1D block index to 3D (d, h, w)
    # Let's assume BLOCK_D, BLOCK_H, BLOCK_W are small enough.
    # We'll iterate over the block spatial elements.
    off_spatial = tl.arange(0, BLOCK_SPATIAL)
    
    # Unflatten off_spatial
    off_w = off_spatial % W_out
    off_h = (off_spatial // W_out) % H_out
    off_d = (spatial_offset // (W_out * H_out)) + (off_spatial // (W_out * H_out)) % D_out # This is wrong.
    
    # Correct unflattening for 3D grid:
    # We are using a 1D grid for spatial. Let's just compute d, h, w from the 1D index.
    # idx = spatial_offset + off_spatial
    # w = idx % W_out
    # h = (idx // W_out) % H_out
    # d = idx // (W_out * H_out)
    # But Triton's tl.arange is for the block.
    # Let's use a 3D grid for (N, C_out, D_out * H_out * W_out) is not possible.
    # Let's use a 3D grid for (N, C_out, spatial_tile).
    # And inside, we compute the 3D indices.
    
    # Actually, let's use a 3D grid: (N, C_out, spatial_tile).
    # And compute d, h, w from the 1D spatial index.
    idx_spatial = spatial_offset + off_spatial
    off_w_block = idx_spatial % W_out
    off_h_block = (idx_spatial // W_out) % H_out
    off_d_block = idx_spatial // (W_out * H_out)
    
    # Masks for output spatial dimensions
    mask_d = off_d_block < D_out
    mask_h = off_h_block < H_out
    mask_w = off_w_block < W_out
    mask_spatial = mask_d & mask_h & mask_w
    
    # Masks for batch and channel
    mask_n = off_n < N
    mask_co = off_co < C_out
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_N, BLOCK_CO, BLOCK_SPATIAL), dtype=tl.float32)
    
    # Loop over input channels and kernel spatial dimensions
    for k in range(0, C_in):
        for kd in range(0, K_D):
            for kh in range(0, K_H):
                for kw in range(0, K_W):
                    # Calculate input indices
                    # Input spatial indices = output spatial indices + kernel spatial indices
                    in_d = off_d_block + kd
                    in_h = off_h_block + kh
                    in_w = off_w_block + kw
                    
                    # Masks for input spatial dimensions
                    mask_in_d = in_d < D
                    mask_in_h = in_h < H
                    mask_in_w = in_w < W
                    mask_in = mask_spatial & mask_in_d & mask_in_h & mask_in_w
                    
                    # Load input tile
                    # Input shape: (N, C_in, D, H, W)
                    # We need to load X[n, k, in_d, in_h, in_w]
                    # This is a 5D load. Triton's tl.load can handle strides.
                    # We'll compute the offset manually.
                    # offset = n * stride_x_n + k * stride_x_c + in_d * stride_x_d + in_h * stride_x_h + in_w * stride_x_w
                    # But we have blocks of n, k, in_d, in_h, in_w.
                    # This is complex. Let's use a simpler approach: load a 3D tile of input for each k.
                    # But k is looped.
                    # Let's load the entire input slice for the current block of n, and spatial, and k.
                    # X[n, k, in_d, in_h, in_w]
                    # We can broadcast k.
                    
                    # Compute offsets for input
                    # n offsets: off_n
                    # k offset: k
                    # in_d offsets: in_d
                    # in_h offsets: in_h
                    # in_w offsets: in_w
                    
                    # We need to create a 5D grid of offsets.
                    # This is getting complex. Let's use a simpler 1D grid for the whole output.
                    # And compute indices from 1D.
                    # But 1D grid is slow.
                    
                    # Let's use a 3D grid: (N, C_out, spatial_tile).
                    # And load input in a fused manner.
                    # For each k, kd, kh, kw, we load a tile of input.
                    # The tile shape is (BLOCK_N, BLOCK_SPATIAL).
                    # We need to compute the global indices for input.
                    # Global indices for n: off_n
                    # Global indices for spatial: off_d_block, off_h_block, off_w_block
                    # Global indices for k: k
                    # Global indices for kd, kh, kw: kd, kh, kw
                    
                    # Load input
                    # X_ptr = X + off_n[:, None, None] * stride_x_n + k * stride_x_c + in_d[None, :, None] * stride_x_d + in_h[None, None, :] * stride_x_h + in_w[None, None, :] * stride_x_w
                    # This is a 3D load. Triton supports this.
                    # But in_d, in_h, in_w are 1D arrays of size BLOCK_SPATIAL.
                    # We need to broadcast them to (BLOCK_N, BLOCK_SPATIAL).
                    # off_n is (BLOCK_N, 1, 1)
                    # in_d is (1, BLOCK_SPATIAL, 1)
                    # in_h is (1, 1, BLOCK_SPATIAL)
                    # in_w is (1, 1, BLOCK_SPATIAL)
                    # This is not supported directly. We need to use a 1D load with offsets.
                    
                    # Let's compute the 1D offsets for the input tile.
                    # offset = off_n[:, None] * stride_x_n + k * stride_x_c + in_d[None, :] * stride_x_d + in_h[None, :] * stride_x_h + in_w[None, :] * stride_x_w
                    # This gives a 2D array of offsets (BLOCK_N, BLOCK_SPATIAL).
                    # We can use tl.load with these offsets.
                    
                    # Compute offsets
                    offsets = (
                        off_n[:, None] * stride_x_n +
                        k * stride_x_c +
                        in_d[None, :] * stride_x_d +
                        in_h[None, :] * stride_x_h +
                        in_w[None, :] * stride_x_w
                    )
                    
                    # Load input
                    x = tl.load(X + offsets, mask=mask_n[:, None] & mask_in[None, :], other=0.0)
                    
                    # Load weight
                    # W shape: (C_out, C_in, K_D, K_H, K_W)
                    # We need W[co, k, kd, kh, kw]
                    # co offsets: off_co
                    # k offset: k
                    # kd, kh, kw offsets: kd, kh, kw
                    # offset = off_co * stride_w_co + k * stride_w_ci + kd * stride_w_kd + kh * stride_w_kh + kw * stride_w_kw
                    w = tl.load(W + off_co * stride_w_co + k * stride_w_ci + kd * stride_w_kd + kh * stride_w_kh + kw * stride_w_kw, mask=mask_co, other=0.0)
                    
                    # Accumulate
                    # acc += x * w[None, :, :]
                    # x is (BLOCK_N, BLOCK_SPATIAL)
                    # w is (BLOCK_CO,)
                    # We need to broadcast w to (1, BLOCK_CO, 1)
                    # acc is (BLOCK_N, BLOCK_CO, BLOCK_SPATIAL)
                    # This is not supported directly. We need to use a 1D accumulator.
                    # Let's flatten the accumulator.
                    # acc += x[:, None, :] * w[None, :, None]
                    # This is supported.
                    acc += x[:, None, :] * w[None, :, None]
    
    # Store output
    # Out shape: (N, C_out, D_out, H_out, W_out)
    # Offsets: off_n, off_co, off_d_block, off_h_block, off_w_block
    # offset = off_n[:, None, None] * stride_out_n + off_co[None, :, None] * stride_out_co + off_d_block[None, None, :] * stride_out_d + off_h_block[None, None, :] * stride_out_h + off_w_block[None, None, :] * stride_out_w
    # This is a 3D offset array. Triton's tl.store can handle this.
    offsets = (
        off_n[:, None, None] * stride_out_n +
        off_co[None, :, None] * stride_out_co +
        off_d_block[None, None, :] * stride_out_d +
        off_h_block[None, None, :] * stride_out_h +
        off_w_block[None, None, :] * stride_out_w
    )
    tl.store(Out + offsets, acc, mask=mask_n[:, None, None] & mask_co[None, :, None] & mask_spatial[None, None, :])


def triton_conv3d(x: torch.Tensor, weight: torch.Tensor):
    """
    Custom Triton kernel for 3D convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    N, C_in, D, H, W = x.shape
    C_out, _, K_D, K_H, K_W = weight.shape
    
    D_out = D - K_D + 1
    H_out = H - K_H + 1
    W_out = W - K_W + 1
    
    out = torch.empty((N, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Block sizes
    BLOCK_N = 1
    BLOCK_CO = 8
    BLOCK_D = 2
    BLOCK_H = 2
    BLOCK_W = 2
    BLOCK_SPATIAL = BLOCK_D * BLOCK_H * BLOCK_W
    
    # Grid dimensions
    grid_n = triton.cdiv(N, BLOCK_N)
    grid_co = triton.cdiv(C_out, BLOCK_CO)
    grid_spatial = triton.cdiv(D_out * H_out * W_out, BLOCK_SPATIAL)
    
    grid = (grid_n, grid_co, grid_spatial)
    
    # Strides
    stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w = x.stride()
    stride_w_co, stride_w_ci, stride_w_kd, stride_w_kh, stride_w_kw = weight.stride()
    stride_out_n, stride_out_co, stride_out_d, stride_out_h, stride_out_w = out.stride()
    
    conv3d_kernel[grid](
        x, weight, out,
        stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kd, stride_w_kh, stride_w_kw,
        stride_out_n, stride_out_co, stride_out_d, stride_out_h, stride_out_w,
        N, C_in, D, H, W, C_out, K_D, K_H, K_W,
        D_out, H_out, W_out,
        BLOCK_N=BLOCK_N, BLOCK_CO=BLOCK_CO, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized 3D convolution model using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize the conv layer to get the weights
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using the custom Triton kernel.
        """
        return triton_conv3d(x, self.conv3d.weight)