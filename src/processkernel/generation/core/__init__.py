"""Shared machinery for kernel generation: problems in, kernel files out.

Split so the parts that are pure logic (prompt assembly, code extraction, the output
layout) can be tested on a login node against a scripted
:class:`~processkernel.generation.core.backend.Backend`.

Two properties are load-bearing and easy to destroy by accident:

* **One prompt per sample slot, always ``n=1``.** Each slot keeps its own prompt, trace
  and output file.
* **One batch.** Every slot across every problem goes into ONE ``generate`` call; a
  per-problem loop pays vLLM's scheduling cost once per problem.
"""

from __future__ import annotations
