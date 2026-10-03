import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------------------
# Triton kernel: fused AdaptiveMaxPool + flatten + concatenation
# -------------------------------------------------------------
@triton.jit
def adaptive_max_pool2d_fused_kernel(
    inp_ptr,               # input tensor pointer (N, C, H, W)
    out_ptr,               # output tensor pointer (N, total_features)
    N, C, H, W,            # input dimensions
    level,                 # current pyramid level (output height = width = level)
    base_offset,           # offset inside the concatenated output for this level
    total_feat,            # total number of features per sample after concat
    BLOCK_SIZE: tl.constexpr,
):
    # ------------------------------------------------------------------
    # Compute a linear index for each thread (one output element per thread)
    # ------------------------------------------------------------------
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (N * C * level * level)

    # Decode linear index into (n, c, oh, ow)
    n = offs // (C * level * level)
    rem = offs % (C * level * level)
    c = rem // (level * level)
    rem2 = rem % (level * level)
    oh = rem2 // level
    ow = rem2 % level

    # ------------------------------------------------------------------
    # Compute the pooling region (adaptive max pooling semantics)
    # ------------------------------------------------------------------
    h_start = (oh * H) // level
    h_end   = ((oh + 1) * H + level - 1) // level   # ceil
    w_start = (ow * W) // level
    w_end   = ((ow + 1) * W + level - 1) // level   # ceil

    # Initialise max value
    max_val = tl.full([BLOCK_SIZE], -3.4028235e38, dtype=tl.float32)

    # Iterate over the region (dynamic loops are supported in Triton)
    hi = h_start
    while hi < h_end:
        wi = w_start
        while wi < w_end:
            # Compute flat input offset for each thread
            inp_offset = ((n * C + c) * H + hi) * W + wi
            val = tl.load(inp_ptr + inp_offset, mask=mask, other=-3.4028235e38)
            max_val = tl.maximum(max_val, val)
            wi += 1
        hi += 1

    # ------------------------------------------------------------------
    # Write the result into the correct place of the concatenated output
    # ------------------------------------------------------------------
    # Position inside the per‑sample feature vector
    level_feat_offset = c * level * level + oh * level + ow
    out_offset = n * total_feat + base_offset + level_feat_offset
    tl.store(out_ptr + out_offset, max_val, mask=mask)


def triton_spp_forward(x: torch.Tensor, levels):
    """
    Performs Spatial Pyramid Pooling using a fused Triton kernel.
    Returns a tensor of shape (N, C * sum(l*l for l in levels)).
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    x = x.contiguous()
    N, C, H, W = x.shape
    total_feat = C * sum(l * l for l in levels)
    out = torch.empty(N, total_feat, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128  # can be tuned

    # Levels must be sorted to compute correct base offsets
    sorted_levels = sorted(levels)
    acc = 0
    for level in sorted_levels:
        base_offset = C * acc                     # offset in the concatenated vector
        acc += level * level                      # accumulate feature count for next level

        total_threads = N * C * level * level
        grid = lambda meta: ((total_threads + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        adaptive_max_pool2d_fused_kernel[grid](
            x,
            out,
            N, C, H, W,
            level,
            base_offset,
            total_feat,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return out


# -------------------------------------------------------------
# Optimized Model
# -------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, levels=[1, 2, 4]):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        # x expected shape: (N, C, H, W)
        return triton_spp_forward(x, self.levels)


# -------------------------------------------------------------
# Compatibility with the original interface
# -------------------------------------------------------------
def get_inputs():
    # Example input matching the original benchmark
    return [torch.rand([4, 4, 4, 4]).cuda()]

def get_init_inputs():
    return [[], {}]

# expose the model name expected by the benchmark harness
Model = ModelNew