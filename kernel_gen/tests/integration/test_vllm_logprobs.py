"""``_topk`` against vLLM's real logprob containers, in a subprocess.

The contract being checked is not ours. vLLM hands back logprobs in two shapes -- one
dict per position, or ``FlatLogprobs``'s parallel primitive lists -- and both are read
by ``_topk`` in ways that depend on undocumented details: that the *sampled* token is
inserted before the top-K, that a row is one entry wider when the sample ranked outside
it, and that the flat container's ``start_indices``/``end_indices`` bound each position.
A hand-written stand-in would encode our belief about all three and would go on passing
after vLLM changed any of them, which is the one failure this file exists to catch.

**Why a subprocess.** Importing vLLM starts threads, and ``checker/scan.py`` uses
``mp.get_context("fork")`` -- deliberately, since the scanner never imports vLLM. But a
pytest process that has imported vLLM then forks from a multi-threaded parent, which is
a genuine deadlock risk and shows up as a DeprecationWarning in ``test_scan``. Keeping
the import out of the shared process costs one subprocess and removes the hazard.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# this file → integration → tests → kernel_gen → repo root (four levels up)
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Two positions, built exactly as vLLM's engine builds them (see
# LogprobsProcessor._update_sample_logprobs -> append_logprobs_for_next_position):
# the sampled token first with its true rank, then the top-K at ranks 1..K.
#   position 0: sampled id 3, which also sits at rank 2 of the top-2 -> row dedupes to 2
#   position 1: sampled id 99, which ranked 5th, outside the top-2 -> row is 3 wide
SCRIPT = """
import sys
sys.path.insert(0, %(repo)r)
from vllm.logprobs import append_logprobs_for_next_position, create_sample_logprobs
from kernel_gen.core.backend import _topk
from kernel_gen.core.trace import pack

for flat in (False, True):
    container = create_sample_logprobs(flat)
    append_logprobs_for_next_position(container, [3, 7, 3], [-1.2, -0.2, -1.2], [None]*3, 2, 2)
    append_logprobs_for_next_position(container, [99, 1, 2], [-9.0, -0.1, -2.0], [None]*3, 5, 2)

    rows = _topk(container)
    assert len(rows) == 2, (flat, rows)
    assert dict(rows[0]) == {3: -1.2, 7: -0.2}, (flat, rows[0])
    assert dict(rows[1]) == {99: -9.0, 1: -0.1, 2: -2.0}, (flat, rows[1])

    trace = pack([3, 99], rows, k=2)
    assert trace.topk_ids[0].tolist() == [7, 3], trace.topk_ids[0]   # sorted, not as given
    assert trace.sampled_rank.tolist() == [2, 3], trace.sampled_rank
    assert abs(trace.sampled_lp[1] + 9.0) < 1e-4, trace.sampled_lp   # kept past truncation
    assert 99 not in trace.topk_ids

assert _topk(None) is None

# The flat container must be read through its lists, never iterated or indexed --
# both of those rebuild the per-position dicts that asking for it was meant to avoid.
from vllm.logprobs import FlatLogprobs
class Tripwire(FlatLogprobs):
    def __iter__(self): raise AssertionError("_topk iterated the flat container")
    def __getitem__(self, index): raise AssertionError("_topk indexed the flat container")

source = create_sample_logprobs(True)
append_logprobs_for_next_position(source, [3, 7], [-1.2, -0.2], [None]*2, 1, 2)
assert len(_topk(Tripwire(**vars(source)))) == 1

print("CONTRACT OK")
""" % {"repo": REPO_ROOT}


def test_topk_reads_both_real_vllm_containers():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600
    )
    if "ModuleNotFoundError: No module named 'vllm'" in result.stderr:
        pytest.skip("vllm not installed")
    assert result.returncode == 0, result.stderr[-3000:]
    assert "CONTRACT OK" in result.stdout
