"""Render the round-2 feedback blocks. Pure text; no GPU, no model.

This module is the experiment. A3 and A4 are handed the *same* seed kernel with the *same*
sampling budget; the only thing that differs is the text these two functions produce. So any
measured difference between the arms is attributable to the feedback and nothing else.

A3 gets what you would naturally tell a model that wrote a slow kernel: how slow it was.

A4 gets the tuning result -- and the point is that a tuning result is not a *number*, it is a
*diagnosis*. The winning config alone would be nearly useless: told "use BLOCK_SIZE=1024",
the model can only copy it, and copying it reproduces exactly what the sweep already did for
free. What can't be copied is the shape of the response surface:

  * latency flat across every config  -> the knobs are irrelevant; the kernel is slow for a
    structural reason and no constant will save it. Rewrite the algorithm.
  * latency falling monotonically to the edge of the grid -> the search wanted to keep going;
    the kernel is starved on memory traffic.
  * some configs producing WRONG RESULTS -> the kernel's correctness silently depends on the
    constant it happened to pick. That is a latent bug, and it is a bug report the model can
    act on regardless of speed.

So we hand over the whole table, not the winner.
"""

from __future__ import annotations

from autotune.knobs import LAUNCH_KNOBS


def select_seed(sweep_summary: dict, problem_id: int) -> dict | None:
    """The A2 champion for a problem: the sample whose *tuned* runtime is lowest.

    Both arms get this same kernel. It is chosen on tuned runtime rather than as-generated
    runtime because that is the kernel A2 would actually hand you, which is what makes
    "A4 vs A2" read as: does one more LLM round improve on what the tuner already gave you?
    """
    candidates = [
        {"kernel": kernel, **entry}
        for kernel, entry in sweep_summary.items()
        if entry.get("problem_id") == problem_id and entry.get("best_ms")
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e["best_ms"])


def build_timing_feedback(seed: dict, baseline_ms: float) -> str:
    """Arm A3: the seed's own runtime, and nothing else.

    Deliberately quotes the *as-generated* time, not the tuned one, so no tuning information
    leaks into the control arm.
    """
    ms = seed["identity_ms"] or seed["best_ms"]
    speedup = baseline_ms / ms if ms else float("nan")
    return f"""## Feedback on your previous solution

Your kernel above is CORRECT.

  Measured latency:        {ms:.4f} ms  (mean of 100 timed runs)
  PyTorch eager baseline:  {baseline_ms:.4f} ms
  Current speedup:         {speedup:.2f}x

Write an improved version of this kernel that runs faster while remaining correct.
Output the complete Python code in a ```python block."""


def _table(seed: dict, max_rows: int = 26) -> str:
    """The config-to-latency table, correct rows fastest-first, then the failures."""
    rows = seed.get("table", [])[:max_rows]
    identity = seed.get("identity_config") or {}
    knob_names = sorted(
        {k for r in rows for k in r["config"] if k not in LAUNCH_KNOBS}
        | {k for k in identity if k not in LAUNCH_KNOBS}
    )
    # Only show a launch knob if we actually varied it, or the kernel itself set it. A 1-D
    # kernel never sweeps num_stages, so an all-"-" column would be pure noise in the context.
    cols = knob_names + [
        k for k in LAUNCH_KNOBS
        if any(k in r["config"] for r in rows)
        or (identity.get(k, "default") != "default")
    ]
    if not cols:
        return "(no tunable constants found in your kernel)"

    header = "| " + " | ".join(cols) + " | latency (ms) | status |"
    rule = "|" + "|".join(["---:"] * len(cols)) + "|---:|:---|"
    lines = [header, rule]
    for r in rows:
        # Config 0 is the model's own constants: its config dict is empty (we patch nothing),
        # so fill the row from the values we read out of its source.
        source = identity if r["config_id"] == 0 else r["config"]
        cells = [str(source.get(c, "-")) for c in cols]
        ms = f"{r['runtime_ms']:.4f}" if r.get("runtime_ms") else "–"
        status = "ok" if r["status"] == "ok" else r["status"].upper().replace("_", " ")
        if r["config_id"] == 0:
            status += "  <-- YOUR ORIGINAL CONSTANTS"
        lines.append("| " + " | ".join(cells) + f" | {ms} | {status} |")
    return "\n".join(lines)


def build_tuning_feedback(seed: dict, baseline_ms: float) -> str:
    """Arm A4: the whole response surface, plus what it implies."""
    identity = seed["identity_ms"] or seed["best_ms"]
    best, gain = seed["best_ms"], seed.get("tuning_gain") or 1.0
    speedup = baseline_ms / identity if identity else float("nan")

    notes = []
    if seed.get("at_grid_edge"):
        notes.append(
            "- The best value sits at the EDGE of the range we swept: we never tried larger "
            "(or smaller) values, so the true optimum may lie beyond it. A kernel that keeps "
            "wanting bigger tiles is usually starved on memory traffic."
        )
    if seed.get("n_wrong_result", 0) > 0:
        notes.append(
            f"- {seed['n_wrong_result']} configuration(s) produced a WRONG RESULT. Your "
            "kernel's correctness silently depends on the constants you happened to choose "
            "-- most likely a missing or incorrect bounds mask, or an assumption that a "
            "dimension divides evenly by the block size. Fix that."
        )
    if gain < 1.05:
        notes.append(
            "- Tuning the constants changed the runtime by less than 5%. The launch "
            "configuration is NOT your bottleneck: no choice of block size or warp count "
            "will make this kernel fast. The problem is structural -- look at the algorithm, "
            "the memory access pattern, and how much work each program does."
        )

    notes_block = ("\n" + "\n".join(notes) + "\n") if notes else ""

    return f"""## Feedback on your previous solution

Your kernel above is CORRECT.

  Latency with your constants:  {identity:.4f} ms
  PyTorch eager baseline:       {baseline_ms:.4f} ms  (your speedup: {speedup:.2f}x)

We then automatically swept its launch configuration on the target GPU and measured every
combination. Full results:

{_table(seed)}

  Best configuration found:  {_fmt_config(seed.get('best_config'))}
  Best latency:              {best:.4f} ms  ({gain:.2f}x faster than your own constants)
{notes_block}
IMPORTANT -- how to use this:

After you submit, the constants (BLOCK_SIZE / BLOCK_SIZE_M / TILE_* / num_warps /
num_stages) will AGAIN be tuned automatically over a grid. So do NOT spend any effort
choosing them, and do NOT hardcode the best configuration above. Keep the tile sizes as
`tl.constexpr` parameters exactly as you did before -- a kernel with the constants baked in
scores worse, not better.

The table is there to tell you WHERE THE KERNEL IS BOTTLENECKED, not which number to paste.
Use it to change the kernel's STRUCTURE: memory access patterns and coalescing, fusing
operations to avoid round-trips to global memory, eliminating redundant loads and stores,
how work is partitioned across programs, and vectorization.

Output the complete Python code in a ```python block."""


def _fmt_config(config: dict | None) -> str:
    if not config:
        return "(your original constants were already the best we tried)"
    return ", ".join(f"{k}={v}" for k, v in sorted(config.items()))


def build_feedback(arm: str, seed: dict, baseline_ms: float) -> str:
    if arm == "timing":
        return build_timing_feedback(seed, baseline_ms)
    if arm == "tuning":
        return build_tuning_feedback(seed, baseline_ms)
    raise ValueError(f"unknown arm {arm!r}")
