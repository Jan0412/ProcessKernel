"""The submission analyzer, and the guard against the bug the Registry replaced.

The module-global ``CHECKS`` list this package's registry replaced would have been shared:
importing ``checker.submission`` would have added S1.* to the linter's collection, so every
lint call would have run them and every lint report would have carried ids that answer a
different question. That failure would be silent, so it gets its own test.
"""

from __future__ import annotations

import checker
from checker.lint import LintAnalyzer
from checker.lint.checks import LINT_REGISTRY
from checker.submission import SubmissionAnalyzer
from checker.submission.checks import SUBMISSION_REGISTRY

PRELUDE = "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n"

CHEATING = PRELUDE + '''

@triton.jit
def k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pass


class ModelNew(nn.Module):
    def forward(self, x):
        return torch.conv2d(x, x)
'''

BROKEN = PRELUDE + "def k(x, W, W):\n    return x\n"


class TestRegistriesAreIndependent:
    def test_the_lint_registry_still_holds_exactly_the_eleven_checks(self):
        assert LINT_REGISTRY.check_ids == [
            "F1.1", "F1.2", "F1.3", "F1.4", "F1.5", "F1.6", "F1.7",
            "F2.1", "F2.2", "F2.3", "F2.4",
        ]

    def test_the_submission_registry_holds_only_its_own(self):
        assert SUBMISSION_REGISTRY.check_ids == ["S1.0", "S1.1", "S1.2", "S1.3"]

    def test_they_are_not_the_same_object(self):
        assert LINT_REGISTRY is not SUBMISSION_REGISTRY
        assert LINT_REGISTRY.checks is not SUBMISSION_REGISTRY.checks

    def test_the_linter_never_emits_a_submission_finding(self):
        report = checker.analyze_source(BROKEN, "<test>")
        assert not any(f.check_id.startswith("S1.") for f in report.findings)

    def test_the_submission_gate_never_emits_a_lint_finding(self):
        # The cheating kernel is full of F1.* findings; the gate has no opinion on them.
        report = SubmissionAnalyzer().analyze(CHEATING, "<test>")
        assert not any(f.check_id.startswith(("F1.", "F2.")) for f in report.findings)


class TestShortCircuit:
    def test_a_non_compilable_file_reports_only_the_cause(self):
        """Complaining about a missing ModelNew on a file that will not compile reports a
        consequence instead of the cause, and sends the model chasing the wrong thing."""
        report = SubmissionAnalyzer().analyze(BROKEN, "<test>")

        assert [f.check_id for f in report.findings] == ["S1.0"]

    def test_a_file_that_does_not_even_parse_is_still_reported(self):
        # The linter stays silent on these -- with no tree there is no evidence of any
        # anti-pattern. For this gate, an unparseable file is the whole point.
        report = SubmissionAnalyzer().analyze("def (:\n", "<test>")

        assert [f.check_id for f in report.findings] == ["S1.0"]

    def test_a_loadable_kernel_produces_no_findings(self):
        report = SubmissionAnalyzer().analyze(CHEATING, "<test>")

        assert report.findings == []
        assert report.summary["n_fail"] == 0


class TestSummary:
    def test_the_summary_carries_only_the_generic_counts(self):
        """fallback_ops and wasted_bytes_lower_bound are lint-derived and can never be
        non-zero here; carrying them would be a column that means nothing."""
        summary = SubmissionAnalyzer().analyze(BROKEN, "<test>").summary

        assert summary["n_fail"] == 1
        assert summary["check_ids"] == ["S1.0"]
        assert "fallback_ops" not in summary
        assert "wasted_bytes_lower_bound" not in summary

    def test_the_lint_summary_still_carries_its_derived_keys(self):
        summary = checker.analyze_source(CHEATING, "<test>").summary

        assert "fallback_ops" in summary
        assert "wasted_bytes_lower_bound" in summary


class TestStaticGuarantee:
    def test_analysis_is_pure_ast(self):
        """The whole package must stay importable and runnable on a login node: no torch,
        no triton, no GPU. A careless import silently destroys that."""
        import sys

        before = set(sys.modules)
        SubmissionAnalyzer().analyze(CHEATING, "<test>")
        LintAnalyzer().analyze(CHEATING, "<test>")

        newly = set(sys.modules) - before
        assert not {m for m in newly if m.split(".")[0] in {"torch", "triton", "numpy"}}
