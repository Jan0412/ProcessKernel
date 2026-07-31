"""Property-based tests for ``extract_code_block`` (Hypothesis).

The extraction bug (KGEN-1) was a property violation: "the result is the model's last
complete ModelNew block" failed on real multi-block completions. These generate
multi-block completions and assert the properties that must hold for *all* of them, so
the whole class is under test rather than the handful of examples I happened to write.
"""

from __future__ import annotations

import ast

from hypothesis import given
from hypothesis import strategies as st

from kernel_gen.core.text import extract_code_block

# Block bodies that all parse as Python: some define ModelNew, some are bare fragments.
MODELNEW_BLOCKS = [
    "import torch\n\n\nclass ModelNew:\n    pass",
    "import torch\n\n\nclass ModelNew:\n    def forward(self, x):\n        return x",
    "import torch\nimport triton\n\n\nclass ModelNew:\n    def forward(self, x):\n        return x + 1",
]
FRAGMENT_BLOCKS = [
    "x = tl.load(p)",
    "acc = 0.0\nfor i in range(4):\n    acc += i",
    "m = ModelNew()\nprint(m)",  # a usage example: parses, mentions ModelNew, is not one
]
block_body = st.sampled_from(MODELNEW_BLOCKS + FRAGMENT_BLOCKS)
prose = st.text(alphabet=st.characters(blacklist_characters="`"), min_size=0, max_size=25)


def _completion(bodies: list[str], proses: list[str]) -> str:
    parts = []
    for body, pr in zip(bodies, proses):
        parts.append(f"{pr}\n```python\n{body}\n```")
    return "\n\n".join(parts)


@given(
    bodies=st.lists(block_body, min_size=1, max_size=6),
    proses=st.lists(prose, min_size=6, max_size=6),
)
def test_the_result_always_parses(bodies, proses):
    # Every block body parses, so whichever one is chosen, the result must parse. A
    # result that does not parse means the wrong span was cut.
    out = extract_code_block(_completion(bodies, proses))
    ast.parse(out)


@given(
    bodies=st.lists(block_body, min_size=1, max_size=6),
    proses=st.lists(prose, min_size=6, max_size=6),
)
def test_when_a_submission_exists_the_last_one_is_returned(bodies, proses):
    # The property KGEN-1 violated. A "submission" is a block that DEFINES ModelNew (a
    # class statement), not merely mentions it -- so the usage-example fragment does not
    # count.
    def defines_modelnew(b: str) -> bool:
        return "class ModelNew" in b

    if not any(defines_modelnew(b) for b in bodies):
        return  # no submission to prefer; extractor returns its best effort, tested elsewhere

    out = extract_code_block(_completion(bodies, proses))
    last_submission = [b for b in bodies if defines_modelnew(b)][-1]
    assert out.strip() == last_submission.strip()


@given(
    bodies=st.lists(block_body, min_size=1, max_size=4),
    proses=st.lists(prose, min_size=4, max_size=4),
)
def test_extraction_is_idempotent_on_its_own_output(bodies, proses):
    # Extracting an already-extracted kernel (no fences, pure code) returns it unchanged.
    # If it did not, a re-run of any downstream stage would corrupt the file.
    once = extract_code_block(_completion(bodies, proses))
    twice = extract_code_block(once)
    assert once == twice
