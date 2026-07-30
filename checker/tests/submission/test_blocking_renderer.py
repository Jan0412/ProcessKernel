"""The blocking renderer.

Returning ``None`` is the load-bearing case: it is what tells the critic the file is
loadable and the linter's feedback should be shown instead. Getting it wrong either blocks
every kernel forever or blocks none.
"""

from __future__ import annotations

import pytest

from checker.core.feedback import Renderer
from checker.submission import SubmissionAnalyzer
from checker.submission.feedback import BlockingRenderer

PRELUDE = "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n"

GOOD = PRELUDE + '''

class ModelNew(nn.Module):
    def forward(self, x):
        return x
'''

DUPLICATE_ARG = PRELUDE + "def k(x, W, W):\n    return x\n" + '''

class ModelNew(nn.Module):
    def forward(self, x):
        return x
'''

MISSING_NN = "import torch\n" + '''

class ModelNew(nn.Module):
    def forward(self, x):
        return x
'''


def render(source: str, previous: set[str] | None = None) -> str | None:
    report = SubmissionAnalyzer().analyze(source, "<test>")
    return BlockingRenderer().render(report, previous)


def test_it_is_a_renderer():
    assert isinstance(BlockingRenderer(), Renderer)


def test_a_loadable_file_renders_nothing():
    # None is the signal that the critic should fall through to the lint feedback.
    assert render(GOOD) is None


def test_a_non_compilable_file_renders_the_reason():
    text = render(DUPLICATE_ARG)

    assert text is not None
    assert "could not be loaded" in text
    assert "S1.0" in text
    assert "duplicate argument" in text


def test_the_duplicated_name_reaches_the_prompt():
    """The point of the whole change: the model is told *which* parameter, not just that
    something is broken."""
    assert "W" in render(DUPLICATE_ARG)


def test_a_missing_import_names_the_alias():
    text = render(MISSING_NN)

    assert "S1.3" in text
    assert "`nn`" in text


def test_it_asks_for_a_complete_file():
    # A diff cannot be evaluated; every round must produce a whole file.
    assert "COMPLETE corrected file" in render(DUPLICATE_ARG)


def test_it_says_nothing_else_was_checked():
    """Honest framing: the gate really did stop before the linter ran, and claiming a
    clean bill of health on the rest would be a lie."""
    assert "nothing else about the kernel was checked" in render(DUPLICATE_ARG)


def test_a_repeat_finding_is_marked():
    text = render(DUPLICATE_ARG, previous={"S1.0"})

    assert "still here" in text


def test_a_first_time_finding_is_not_marked():
    assert "still here" not in render(DUPLICATE_ARG)


@pytest.mark.parametrize("source", [GOOD, DUPLICATE_ARG, MISSING_NN])
def test_rendering_never_raises(source):
    BlockingRenderer().render(SubmissionAnalyzer().analyze(source, "<test>"))
