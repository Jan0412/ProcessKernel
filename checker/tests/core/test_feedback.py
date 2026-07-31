"""The findings -> prompt renderer.

Two behaviors here decide whether the whole loop works or quietly wastes GPU hours:
``render`` returning None (that is the early stop -- get it wrong and every sample runs
every round), and fails suppressing warns (get it wrong and the model is told to
optimize a kernel that is secretly calling torch).
"""

from __future__ import annotations

import re

import pytest

from checker import analyze_source
from checker.core.feedback import (
    Feedback,
    Renderer,
    StagedRenderer,
    actionable,
    render,
)
from checker.core.model import FileReport, Finding

from helpers import PRELUDE, forward_with

# A kernel that is defined but never launched, and whose forward hands the real work
# back to PyTorch: F1.2 + F1.4, both fail-severity.
CHEATING = PRELUDE + '''
class ModelNew(nn.Module):
    def forward(self, x):
        return F.softmax(torch.relu(x), dim=-1)


def get_inputs():
    return [torch.randn(1024, 1024)]
'''

# Launches the kernel and returns its output: nothing for the linter to say.
HONEST = PRELUDE + forward_with("out")


def _finding(check_id: str, severity: str) -> Finding:
    return Finding(check_id=check_id, severity=severity, message=f"{check_id} happened.")


def _report(*findings: Finding, parse_status="ok", notes=None) -> FileReport:
    summary = {"notes": notes} if notes else {}
    return FileReport(
        path="k.py", parse_status=parse_status, findings=list(findings), summary=summary
    )


# -- the early stop --------------------------------------------------------


def test_a_clean_kernel_renders_to_none():
    # This None is what makes a clean sample cost nothing for the rest of the run.
    assert render(analyze_source(HONEST, "k.py")) is None


def test_a_dirty_kernel_renders_its_findings():
    text = render(analyze_source(CHEATING, "k.py"))

    assert text is not None
    assert "F1.2" in text and "F1.4" in text
    assert "```python" in text  # tells the model what shape the answer must take


def test_info_findings_are_never_actionable():
    # Nothing is asked of the model, so an info must not keep a slot alive for a round.
    assert render(_report(_finding("F9.9", "info"))) is None


# -- severity staging ------------------------------------------------------


def test_fails_suppress_warns():
    report = _report(_finding("F1.4", "fail"), _finding("F2.1", "warn"))
    text = render(report)

    assert "F1.4" in text
    # Advice about memory traffic is worse than useless on a kernel that is cheating.
    assert "F2.1" not in text
    assert "Performance" not in text


def test_warns_are_shown_once_there_are_no_fails():
    text = render(_report(_finding("F2.1", "warn"), _finding("F2.3", "warn")))

    assert "F2.1" in text and "F2.3" in text
    assert "Performance" in text


def test_the_policy_is_ablatable():
    report = _report(_finding("F1.4", "fail"), _finding("F2.1", "warn"))

    assert [f.check_id for f in actionable(report, "severity")] == ["F1.4"]
    assert [f.check_id for f in actionable(report, "fails-only")] == ["F1.4"]
    assert [f.check_id for f in actionable(report, "all")] == ["F1.4", "F2.1"]
    # fails-only means a kernel with nothing but warns is already done.
    assert render(_report(_finding("F2.1", "warn")), policy="fails-only") is None


# -- the cap ---------------------------------------------------------------


def test_the_cap_drops_performance_advice_before_it_drops_a_correctness_bug():
    report = _report(
        *[_finding(f"F1.{i}", "fail") for i in range(1, 4)],
        *[_finding(f"F2.{i}", "warn") for i in range(1, 5)],
    )
    text = render(report, max_findings=3, policy="all")

    assert all(f"F1.{i}" in text for i in range(1, 4))
    assert "F2.1" not in text
    assert "4 further finding(s) omitted" in text


# -- history ---------------------------------------------------------------


def test_a_finding_that_survived_a_round_is_marked_as_such():
    report = _report(_finding("F1.4", "fail"), _finding("F1.2", "fail"))
    text = render(report, previous_check_ids={"F1.4"})

    marked = [line for line in text.splitlines() if "still here" in line]
    assert len(marked) == 1
    assert "F1.4" in marked[0]  # only the one that survived; F1.2 is new this round


# -- the file that does not parse ------------------------------------------


def test_a_syntax_error_renders_its_own_block():
    # Today this file is written to disk and silently scores zero at eval. In a loop it
    # is the cheapest thing in the world to fix.
    report = analyze_source("def forward(x:\n", "k.py")
    assert report.parse_status == "syntax_error"

    text = render(report)
    assert text is not None
    assert "not valid Python" in text
    assert "from scratch" in text


def test_an_empty_generation_renders_its_own_block():
    text = render(_report(parse_status="empty"))
    assert "no code at all" in text


# -- the renderer as an object ---------------------------------------------
# `render` is the free function the critic has always called; StagedRenderer is the same
# behaviour as a Renderer, so the critic can compose it with the submission gate's
# renderer without knowing which one it holds.


def test_the_staged_renderer_is_a_renderer():
    assert isinstance(StagedRenderer(), Renderer)


def test_the_free_function_and_the_renderer_agree():
    report = analyze_source(CHEATING, "k.py")
    assert StagedRenderer().render(report) == render(report)


def test_a_renderer_must_implement_feedback():
    # `render` is derived, so `feedback` is the one a subclass has to supply.
    class Empty(Renderer):
        pass

    with pytest.raises(TypeError):
        Empty()


def test_the_renderer_carries_its_policy_and_cap():
    report = analyze_source(CHEATING, "k.py")

    assert StagedRenderer(policy="fails-only").render(report) == render(
        report, policy="fails-only"
    )
    assert StagedRenderer(max_findings=1).render(report) == render(report, max_findings=1)


def test_a_clean_report_renders_nothing():
    # The early stop, stated on the object: None is what makes a clean sample free.
    assert StagedRenderer().render(_report()) is None


def test_an_unknown_policy_is_refused():
    """The policy selects what the model is shown, so a typo must not silently degrade to
    "show nothing" and quietly end every slot."""
    with pytest.raises(ValueError, match="unknown feedback policy"):
        actionable(_report(), policy="whatever")


# -- what the text actually contains (KGEN-17/18) ---------------------------
# Two readers downstream -- the repair prompt's repeat marker and the readout's mechanism
# table -- were re-deriving "what was the model told" from a summary that cannot express
# it, because staging and the cap happen here. So the renderer says it outright.


def _ids_in(text: str | None) -> set[str]:
    """The check ids actually bulleted in the prompt, read back out of the text."""
    return set(re.findall(r"^- \*\*([A-Z]\d+\.\d+)\*\*", text or "", re.MULTILINE))


def test_the_ids_reported_are_exactly_the_ids_in_the_text():
    report = _report(
        _finding("F1.4", "fail"),
        _finding("F1.2", "fail"),
        _finding("F2.1", "warn"),
        _finding("F9.9", "info"),
    )
    result = StagedRenderer(policy="all").feedback(report)

    assert result.check_ids == _ids_in(result.text)
    assert result.check_ids == {"F1.4", "F1.2", "F2.1"}  # info is never actionable


def test_a_suppressed_warn_is_not_reported_as_shown():
    """The 15.2% case: severity staging hides every warn while a fail exists, but the
    summary's `check_ids` still lists them -- which is how a round got marked as repeating
    advice the model was never given."""
    report = _report(
        _finding("F1.4", "fail"), _finding("F2.1", "warn"), _finding("F2.3", "warn")
    )
    result = StagedRenderer(policy="severity").feedback(report)

    assert result.check_ids == {"F1.4"}
    assert "F2.1" not in result.text and "F2.3" not in result.text


def test_a_truncated_finding_is_not_reported_as_shown():
    report = _report(*[_finding(f"F1.{i}", "fail") for i in range(1, 4)])
    result = StagedRenderer(max_findings=1).feedback(report)

    assert result.check_ids == _ids_in(result.text)
    assert len(result.check_ids) == 1


def test_a_file_that_does_not_parse_reports_no_ids():
    """Honest, and the reason the empty-source rounds stay the known residual: this block
    names no check, so there is nothing to report."""
    result = StagedRenderer().feedback(_report(parse_status="syntax_error"))

    assert result.text is not None
    assert result.check_ids == frozenset()


def test_a_clean_report_reports_no_text_and_no_ids():
    result = StagedRenderer().feedback(_report())

    assert result.text is None
    assert result.check_ids == frozenset()


@pytest.mark.parametrize("policy", ["severity", "fails-only", "all"])
@pytest.mark.parametrize("cap", [1, 2, 8])
def test_render_still_returns_exactly_what_feedback_says(policy, cap):
    """`render` is derived from `feedback`, so there is exactly one selection path and the
    two can never drift."""
    for source in (CHEATING, HONEST, "def forward(x:\n"):
        report = analyze_source(source, "k.py")
        renderer = StagedRenderer(max_findings=cap, policy=policy)

        assert renderer.render(report) == renderer.feedback(report).text
        assert render(report, max_findings=cap, policy=policy) == renderer.feedback(
            report
        ).text


def test_the_reported_ids_survive_the_repeat_marker():
    # The marker appends to the bullet; the id must still be reported exactly once.
    report = _report(_finding("F1.4", "fail"), _finding("F1.2", "fail"))
    result = StagedRenderer().feedback(report, previous_check_ids={"F1.4"})

    assert result.check_ids == {"F1.4", "F1.2"} == _ids_in(result.text)


def test_feedback_is_hashable_and_frozen():
    # It gets stored on a Review and compared; accidental mutation would desync text/ids.
    result = Feedback(text="x", check_ids=frozenset({"F1.4"}))

    assert hash(result)
    with pytest.raises(Exception):
        result.text = "y"
