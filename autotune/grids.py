"""Which configs to try for a given kernel.

The joint space of every knob is far too large to enumerate: a kernel with three tile
dimensions plus warps and stages has thousands of points, and each one costs a compile and
a benchmark. So we do not cross-product. We pick a bounded, hand-chosen set per shape:

  * 1-D kernels (one BLOCK_SIZE): a full 6x4 sweep of block size against warp count. This
    is the space that actually matters for the elementwise/reduction kernels that dominate
    the corpus, and 24 configs is affordable.
  * tiled kernels (BLOCK_SIZE_M/N/K): the curated ladder below, which walks tile shapes
    from tiny to large with the warp and stage counts that are sane for each. Cross-
    producting M x N x K x warps x stages would be ~500 configs for no benefit -- most of
    them are known-bad (e.g. a 256x256 tile with 1 warp will not fit in registers).

Config 0 is always the identity: the constants the model itself wrote, re-measured inside
the sweep harness. Every tuning gain is a ratio against that, measured the same way, so a
difference in measurement setup cannot masquerade as a speedup.
"""

from __future__ import annotations

from autotune.knobs import TunabilityReport, _role

# 1-D: block size x warps. Powers of two only (tl.arange requires it).
BLOCK_SIZES_1D = (64, 128, 256, 512, 1024, 2048)
NUM_WARPS_1D = (1, 2, 4, 8)

# Tiled: (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages).
# Small tiles get few warps; large tiles get more warps and deeper pipelining.
TILED_LADDER = (
    (16, 16, 16, 2, 2),
    (32, 32, 32, 4, 2),
    (64, 32, 32, 4, 2),
    (32, 64, 32, 4, 2),
    (64, 64, 32, 4, 2),
    (64, 64, 64, 4, 2),
    (128, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (128, 128, 32, 8, 3),
    (128, 64, 64, 8, 3),
    (64, 128, 64, 8, 3),
    (128, 128, 64, 8, 4),
    (128, 256, 64, 8, 4),
    (256, 128, 64, 8, 4),
)

MAX_CONFIGS = 25  # identity + 24


def build_grid(report: TunabilityReport, max_configs: int = MAX_CONFIGS) -> list[dict[str, int]]:
    """Configs to try, identity ({}) first."""
    grid: list[dict[str, int]] = [{}]
    if report.parse_error or not report.n_jit_kernels:
        return grid
    if not report.n_launches:
        # The model wrote a @triton.jit kernel and then never called it -- ModelNew's forward
        # is plain PyTorch. There is no launch to configure, so there is nothing to tune.
        return grid

    roles: dict[str, list[str]] = {}
    for knob in report.knobs:
        roles.setdefault(_role(knob.name), []).append(knob.name)

    if any(r in roles for r in ("M", "N", "K")):
        grid += _tiled(roles, report.has_loop)
    elif roles.get("PLAIN"):
        grid += _one_d(roles["PLAIN"], report)
    else:
        # Launched, but every block size is computed at runtime (triton.next_power_of_2 and
        # friends). We can still ask whether the kernel is warp-sensitive -- cheap, and
        # occasionally worth a lot.
        grid += [{"num_warps": w} for w in (2, 4, 8)]

    return grid[:max_configs]


def _one_d(names: list[str], report: TunabilityReport) -> list[dict[str, int]]:
    """Block size x warps.

    Several distinct plain knobs in one file (rare) are swept locked to a common value.
    Sweeping them independently would square the grid to buy very little: they are almost
    always the block sizes of sibling elementwise kernels with the same access pattern.
    """
    out = []
    for block in BLOCK_SIZES_1D:
        for warps in NUM_WARPS_1D:
            cfg = {name: block for name in names}
            cfg["num_warps"] = warps
            out.append(cfg)
    return out


def _tiled(roles: dict[str, list[str]], has_loop: bool) -> list[dict[str, int]]:
    """Walk the curated tile ladder.

    Knobs we do not have a ladder column for are held at the model's own value:
      * GROUP_SIZE_M -- an L2-swizzle knob, second-order for runtime, and the models
        already tend to write the conventional 8.
      * a plain BLOCK_SIZE living alongside a matmul in the same file -- it belongs to a
        cheap elementwise kernel whose cost is dominated by the matmul we are tuning.
    Sweeping either would multiply the grid for a second-order effect.
    """
    out = []
    for bm, bn, bk, warps, stages in TILED_LADDER:
        cfg: dict[str, int] = {}
        for name in roles.get("M", []):
            cfg[name] = bm
        for name in roles.get("N", []):
            cfg[name] = bn
        for name in roles.get("K", []):
            cfg[name] = bk
        cfg["num_warps"] = warps
        if has_loop:
            # num_stages is software-pipelining depth: it only means anything if there is a
            # loop over K for the compiler to pipeline.
            cfg["num_stages"] = stages
        out.append(cfg)
    return out


def at_grid_edge(config: dict[str, int], report: TunabilityReport) -> bool:
    """Did the winner sit at the boundary of what we tried?

    If so, the search wanted to keep going in that direction and we stopped it -- worth
    telling the model, and worth knowing before we conclude a kernel is tuned out.
    """
    if report.ndim_class == "1d":
        blocks = [v for k, v in config.items() if k not in ("num_warps", "num_stages")]
        if blocks and (min(blocks) == BLOCK_SIZES_1D[0] or max(blocks) == BLOCK_SIZES_1D[-1]):
            return True
        return config.get("num_warps") in (NUM_WARPS_1D[0], NUM_WARPS_1D[-1])
    if report.ndim_class == "tiled":
        key = tuple(
            config.get(n)
            for role in ("M", "N", "K")
            for n in (next((k.name for k in report.knobs if _role(k.name) == role), None),)
            if n
        )
        firsts = {TILED_LADDER[0][:3], TILED_LADDER[-1][:3]}
        return any(key == f[: len(key)] for f in firsts)
    return False
