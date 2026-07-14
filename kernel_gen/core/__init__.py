"""Shared machinery for kernel-generation arms: problems in, kernel files out.

The four ``generate_kernels_*.py`` scripts each carry their own copy of the same
``main()``: load a dataset, build a prompt per problem, sample N completions,
extract the code block, write ``level_{L}_problem_{P}_sample_{S}_kernel.py``. That
shape is fine for a one-shot generation and hopeless for a *loop*, which needs to
re-prompt a subset of samples and stop them individually. It is also untestable --
every one of those scripts needs a GPU to do anything at all.

This package is the loop-capable version of that machinery, split so the parts that
are pure logic (prompt assembly, code extraction, the round bookkeeping) can be
tested on a login node against a scripted :class:`~kernel_gen.core.backend.Backend`.

Two properties are load-bearing and easy to destroy by accident:

* **One prompt per sample slot, always ``n=1``.** A sample is a slot that can stop
  on its own round, so it needs its own prompt. ``n=num_samples`` on a single prompt
  -- what the legacy non-think path does -- cannot express that.
* **Round-major batching.** A round collects every still-active slot across every
  problem into ONE ``generate`` call. The legacy scripts loop per problem; at
  10 samples x 3 rounds the per-problem shape is what makes the loop unaffordable.

A DELIBERATE, TIME-BOXED FORK
-----------------------------
``text.py`` and ``sampling.py`` re-implement ``extract_code_block``,
``problem_id_from_name``, ``parse_problems``, ``load_model`` and the two-pass
think/code sampler from ``generate_kernels_samples.py`` -- roughly 120 duplicated
lines. This is not drift and it is not an oversight.

Pointing ``generate_kernels_samples.py`` at this package would change arm A1's
output: its sampler shape changes (``n=1`` per slot), and it would gain the
``--max-model-len`` flag it currently lacks -- which today silently caps every A1
run at the 16384 default, so it can never emit the 16384 *new* tokens the launch
scripts ask for. That is a real behavior change and it belongs to A1's own port,
with its own smoke run, not to a branch that is supposed to be purely additive to
runs already in flight.

The fork ends when the arms are ported (``arms/samples.py`` etc.), at which point
the duplicated helpers here become the only copy. Until then, ``--dry-run`` prints
the round-0 prompt so it can be diffed against the legacy script's: the two must
stay byte-identical.
"""

from __future__ import annotations
