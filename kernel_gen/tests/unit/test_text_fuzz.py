"""Structural fuzz for ``extract_code_block``: the law the ranking must never break.

**Entry-class monotonicity** -- if any candidate in the completion is a loadable
``ModelNew``, what comes back must be a ``ModelNew``. KGEN-1, 2, 3, 9, 11 and 19 are all
instances of this one property being violated in different ways.

This caught a design that the whole-corpus replay called clean: an early-return branch
that ranked an outside-fence candidate *without* comparing it to the fenced blocks on the
same criteria scored 0 differences over 10,510 real completions and **214 violations in
20,000 generated ones**. The generator builds shapes the model has not happened to emit
yet, which is exactly the gap a frozen corpus cannot cover.

Seeded, so a failure is reproducible. ``FUZZ_TRIALS`` is small enough for CI; raise it
when changing the ranking.
"""

from __future__ import annotations

import ast
import random

import pytest

from kernel_gen.core.text import (
    ENTRY_CLASS,
    _FENCE_TOKEN,
    _fenced_blocks,
    _resume_at_first_statement,
    extract_code_block,
)

FUZZ_TRIALS = 2000
SEED = 20260731

GOOD = (
    "import torch\nimport torch.nn as nn\n\n"
    "class ModelNew(nn.Module):\n    def forward(self, x):\n        return x\n"
)
GOOD_ALT = GOOD.replace("return x", "return x * 2")
TRUNCATED = "import torch\n\nclass ModelNew(nn.Module):\n    def forward(self, x):\n        y = tl.load("
FRAGMENT = "x = tl.load(p)\ny = x + 1\n"
NO_FORWARD = "import torch\nclass ModelNew(nn.Module):\n    def __init__(self):\n        pass\n"
PROSE_ONLY = "Wait, that is wrong.\n"

CODES = [GOOD, GOOD_ALT, TRUNCATED, FRAGMENT, NO_FORWARD, PROSE_ONLY, ""]
TAGS = ["python", "py", "", "cpp", "text"]
PROSE = ["Let me think.\n", "## Plan\n", "Actually wait.\n", "Final:\n", ""]


def _loads(src: str) -> bool:
    try:
        compile(src, "<b>", "exec")
        return True
    except Exception:  # noqa: BLE001
        return False


def _defines_entry(src: str) -> bool:
    try:
        return any(
            isinstance(n, ast.ClassDef) and n.name == "ModelNew" for n in ast.walk(ast.parse(src))
        )
    except Exception:  # noqa: BLE001
        return False


def _completions(rng: random.Random):
    """Prose, fenced blocks (sometimes unterminated) and bare code, in any order."""
    for _ in range(FUZZ_TRIALS):
        parts = []
        for _ in range(rng.randint(1, 6)):
            roll = rng.random()
            if roll < 0.35:
                parts.append(rng.choice(PROSE))
            elif roll < 0.80:
                closing = "```\n" if rng.random() < 0.8 else ""
                parts.append(f"```{rng.choice(TAGS)}\n{rng.choice(CODES)}{closing}")
            else:
                parts.append(rng.choice(CODES))
        yield "".join(parts)


def _reachable_candidates(raw: str) -> list[str]:
    """Everything a correct extractor could have returned.

    The fenced blocks, plus each region *outside* them resumed at its first Python
    statement. Omitting the outside regions is what makes this property vacuous: whole
    ``raw`` never compiles once there is any prose, so a completion whose only answer is
    unfenced would silently look compliant (KGEN-11).
    """
    regions, pos = [], 0
    for m in _FENCE_TOKEN.finditer(raw):
        regions.append(raw[pos:m.start()])
        pos = m.end()
    regions.append(raw[pos:])
    outside = [c for c in (_resume_at_first_statement(r) for r in regions) if c]
    return _fenced_blocks(raw) + outside


@pytest.mark.xfail(
    strict=True,
    reason="KGEN-11: a loadable ModelNew written outside the fenced blocks is unreachable, "
           "because the no-fence fallback only runs when there are no fenced blocks at all. "
           "146/2000 generated completions violate the property. Delete this marker with "
           "the ladder fix.",
)
def test_a_loadable_modelnew_anywhere_is_never_discarded():
    rng = random.Random(SEED)
    violations = []
    for raw in _completions(rng):
        out = extract_code_block(raw)
        if _defines_entry(out):
            continue
        # Nothing was returned that defines ModelNew -- so nothing available may have.
        if any(_loads(c) and _defines_entry(c) for c in _reachable_candidates(raw)):
            violations.append(raw)
    assert not violations, (
        f"{len(violations)}/{FUZZ_TRIALS} completions dropped a loadable ModelNew; "
        f"first:\n{violations[0]!r}"
    )


def test_extraction_never_raises_on_any_generated_shape():
    rng = random.Random(SEED + 1)
    for raw in _completions(rng):
        extract_code_block(raw)  # must not raise


def test_the_result_is_always_derived_from_the_input():
    """Never fabricated text: every non-empty result's first line comes from the input."""
    rng = random.Random(SEED + 2)
    for raw in _completions(rng):
        out = extract_code_block(raw).strip()
        if out:
            assert out.splitlines()[0].strip() in raw, out[:80]


def test_the_entry_class_marker_is_never_invented():
    rng = random.Random(SEED + 3)
    for raw in _completions(rng):
        if ENTRY_CLASS not in raw:
            assert ENTRY_CLASS not in extract_code_block(raw)
