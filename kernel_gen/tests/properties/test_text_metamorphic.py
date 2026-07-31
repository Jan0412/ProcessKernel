"""Metamorphic relations for ``extract_code_block`` (gemtest).

Example-based tests check specific inputs. A metamorphic relation checks how the output
must move when the input is transformed in a known way -- catching bugs on inputs nobody
wrote by hand. The relation here: **prose that contains no code fence cannot change
which kernel is extracted.** Prepending analysis or appending closing remarks is exactly
what real completions do around the code, so if extraction were sensitive to it, every
run would be at the mercy of the model's chattiness.
"""

from __future__ import annotations

import gemtest as gmt

from kernel_gen.core.text import extract_code_block

# Well-formed completions whose extracted kernel is unambiguous and stable. Each is a
# shape seen in real Qwen3.6 output: single block, draft-then-final, plan-then-code.
COMPLETIONS = [
    "```python\nimport torch\n\n\nclass ModelNew:\n    pass\n```",
    (
        "Draft:\n```python\nx = tl.load(p)  # fragment\n```\n\n"
        "Final:\n```python\nimport torch\n\n\nclass ModelNew:\n"
        "    def forward(self, x):\n        return x\n```"
    ),
    (
        "## Plan\nFuse the elementwise ops.\n\n"
        "```python\nimport torch\n\n\nclass ModelNew:\n"
        "    def forward(self, x):\n        return x * 2\n```"
    ),
]

# non-fenced prose: must not introduce a ``` of its own, or it stops being a no-op
_TRAILING = "\n\nThat should give a solid speedup over the baseline.\n"
_LEADING = "First, let me analyze which operators dominate the runtime.\n\n"


trailing_mr = gmt.create_metamorphic_relation(name="trailing_prose_invariance", data=COMPLETIONS)


@gmt.transformation(trailing_mr)
def append_closing_remarks(raw: str) -> str:
    return raw + _TRAILING


@gmt.relation(trailing_mr)
def extraction_unchanged_by_trailing(source_out: str, followup_out: str) -> bool:
    return gmt.equality(source_out, followup_out)


@gmt.system_under_test(trailing_mr)
def test_extract_is_stable_under_trailing_prose(raw: str) -> str:
    return extract_code_block(raw)


leading_mr = gmt.create_metamorphic_relation(name="leading_prose_invariance", data=COMPLETIONS)


@gmt.transformation(leading_mr)
def prepend_analysis(raw: str) -> str:
    return _LEADING + raw


@gmt.relation(leading_mr)
def extraction_unchanged_by_leading(source_out: str, followup_out: str) -> bool:
    return gmt.equality(source_out, followup_out)


@gmt.system_under_test(leading_mr)
def test_extract_is_stable_under_leading_prose(raw: str) -> str:
    return extract_code_block(raw)


# -- KGEN-3: a stray closing fence before the real block is a no-op --------
#
# Reasoning models close an in-reasoning code example with a lone ``` and then write the
# real kernel. That stray closer must not change what is extracted. The old regex read it
# as an opener and paired it with the real block's ```python, swallowing the real block;
# the state-machine parser ignores a bare ``` seen outside a block. This relation checks
# the invariant for EVERY well-formed completion, not just the hand-picked repros.
_STRAY = "Let me sketch the inner loop first.\n```\nThat is the idea.\n\n"

stray_fence_mr = gmt.create_metamorphic_relation(
    name="stray_closing_fence_invariance", data=COMPLETIONS
)


@gmt.transformation(stray_fence_mr)
def prepend_stray_closing_fence(raw: str) -> str:
    return _STRAY + raw


@gmt.relation(stray_fence_mr)
def extraction_unchanged_by_stray_fence(source_out: str, followup_out: str) -> bool:
    return gmt.equality(source_out, followup_out)


@gmt.system_under_test(stray_fence_mr)
def test_extract_is_stable_under_a_stray_closing_fence(raw: str) -> str:
    return extract_code_block(raw)


# -- KGEN-11: an outside-fence answer must not displace a good fenced one -----
#
# This is the 223-case guard as a general law rather than one example. 223 real
# completions carry a loadable ModelNew *outside* the fences AND a correct fenced
# extraction; a rule that preferred the outside one would rewrite every one of them.
# If the gate on `loadable_submissions` ever weakens, every case in COMPLETIONS breaks
# this relation at once.
_OUTSIDE_ANSWER = (
    "\n\nOn reflection, here it is again:\n\n"
    "import torch\n\n\nclass ModelNew:\n    def forward(self, x):\n        return x + 1\n"
)

outside_answer_mr = gmt.create_metamorphic_relation(
    name="outside_answer_does_not_displace_a_good_block", data=COMPLETIONS
)


@gmt.transformation(outside_answer_mr)
def append_an_unfenced_modelnew(raw: str) -> str:
    return raw + _OUTSIDE_ANSWER


@gmt.relation(outside_answer_mr)
def extraction_unchanged_by_outside_answer(source_out: str, followup_out: str) -> bool:
    return gmt.equality(source_out, followup_out)


@gmt.system_under_test(outside_answer_mr)
def test_extract_ignores_an_outside_answer_when_a_good_block_exists(raw: str) -> str:
    return extract_code_block(raw)


# -- fencing the answer is a no-op -------------------------------------------
#
# The same kernel must come back whether the model wrapped its final answer in a fence
# or left it as plain text. Extraction should key on what the code IS, not on whether
# the model remembered the backticks.
_PLAIN = (
    "Draft:\n```python\nx = 1\n```\n\nFinal:\n\n"
    "import torch\n\n\nclass ModelNew:\n    def forward(self, x):\n        return x\n"
)

fencing_mr = gmt.create_metamorphic_relation(name="fencing_the_answer_is_a_noop", data=[_PLAIN])


@gmt.transformation(fencing_mr)
def wrap_the_trailing_answer_in_a_fence(raw: str) -> str:
    head, sep, tail = raw.partition("Final:\n\n")
    return f"{head}{sep}```python\n{tail}```\n"


@gmt.relation(fencing_mr)
def extraction_unchanged_by_fencing(source_out: str, followup_out: str) -> bool:
    return gmt.equality(source_out.strip(), followup_out.strip())


@gmt.system_under_test(fencing_mr)
def test_extract_is_stable_whether_the_answer_is_fenced(raw: str) -> str:
    return extract_code_block(raw)


# -- KGEN-19: corrupting a non-submission cannot change the winner ------------
#
# Entry class is the dominant criterion, so breaking the syntax of a block that has no
# ModelNew must not promote or demote anything. If parseability were ever ranked above
# the entry class, this relation fails.
_JUNK_BLOCK = "\n```python\ny = tl.load(\n```\n"

corrupt_mr = gmt.create_metamorphic_relation(
    name="corrupting_a_non_submission_is_a_noop", data=COMPLETIONS
)


@gmt.transformation(corrupt_mr)
def append_an_unparseable_fragment(raw: str) -> str:
    return raw + _JUNK_BLOCK


@gmt.relation(corrupt_mr)
def extraction_unchanged_by_junk(source_out: str, followup_out: str) -> bool:
    return gmt.equality(source_out, followup_out)


@gmt.system_under_test(corrupt_mr)
def test_extract_ignores_an_appended_unparseable_fragment(raw: str) -> str:
    return extract_code_block(raw)
