"""KGEN-14: "clean" must mean the evaluator can load the file.

``Review.clean`` is the loop's stop signal -- it sets ``traj.done`` and ends the slot. It
used to mean only "the linter had nothing to say", and the linter answers "is this good
Triton?", never "can Python import this and can the benchmark instantiate ModelNew?".
Measured over 9,886 kernels the loop called clean, 367 could not be loaded at all: they
were labelled good, stopped their own repair loop, and scored zero.

These pin the fixed behaviour. The last test in each half is the control -- a gate that
blocks everything would pass the first half and fail the second.
"""

from __future__ import annotations

import pytest

from kernel_gen.core.critics import lint_critic
from kernel_gen.core.model import Problem

PRELUDE = "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n"

KERNEL = '''

@triton.jit
def k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0 + 1.0, mask=mask)
'''

ENTRY = '''

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        k[(1,)](x, out, n, BLOCK=1024)
        return out
'''

VALID = PRELUDE + KERNEL + ENTRY

#: The largest real group: 181 of 217 non-compilable kernels repeat one parameter name.
DUPLICATE_ARG = PRELUDE + '''

@triton.jit
def k(x_ptr, o_ptr, W, W, BLOCK: tl.constexpr):
    pass
''' + ENTRY

#: `nn` used in the class statement, never imported.
MISSING_IMPORT = "import torch\nimport triton\nimport triton.language as tl\n" + KERNEL + ENTRY

NO_ENTRY_CLASS = PRELUDE + KERNEL

NO_FORWARD = PRELUDE + KERNEL + '''

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
'''


@pytest.fixture
def problem(good_kernel_file) -> Problem:
    return Problem(level=1, problem_id=1, name="1_Add.py", ref_arch_src=good_kernel_file)


@pytest.fixture
def critic():
    return lint_critic()


# -- an unloadable file is not clean ----------------------------------------


def test_a_kernel_that_does_not_compile_is_not_clean(critic, problem):
    review = critic(problem, DUPLICATE_ARG, set())

    assert not review.clean
    assert review.text


def test_a_kernel_that_cannot_be_imported_is_not_clean(critic, problem):
    assert not critic(problem, MISSING_IMPORT, set()).clean


def test_a_kernel_with_no_entry_class_is_not_clean(critic, problem):
    assert not critic(problem, NO_ENTRY_CLASS, set()).clean


def test_a_kernel_whose_entry_has_no_forward_is_not_clean(critic, problem):
    assert not critic(problem, NO_FORWARD, set()).clean


def test_a_valid_kernel_is_still_clean(critic, problem):
    """The over-blocking control. A gate that rejected everything would satisfy every
    test above and be worse than the bug."""
    review = critic(problem, VALID, set())

    assert review.clean
    assert review.text == ""


# -- what the model is actually told ----------------------------------------


def test_the_repair_prompt_names_the_duplicated_parameter(critic, problem):
    """"Your file is broken" is not repairable; "duplicate argument 'W'" is."""
    text = critic(problem, DUPLICATE_ARG, set()).text

    assert "W" in text
    assert "duplicate argument" in text


def test_the_repair_prompt_names_the_missing_alias(critic, problem):
    assert "`nn`" in critic(problem, MISSING_IMPORT, set()).text


def test_the_prompt_asks_for_a_complete_file(critic, problem):
    assert "COMPLETE corrected file" in critic(problem, DUPLICATE_ARG, set()).text


# -- staging: a file that will not load gets no performance advice -----------


def test_lint_findings_are_suppressed_while_the_file_is_unloadable(critic, problem):
    """The same argument the linter already makes about fails outranking warns, one level
    up: advice about fusing memory traffic is noise on a file Python cannot import."""
    cheating_and_broken = PRELUDE + '''

@triton.jit
def k(x_ptr, o_ptr, W, W, BLOCK: tl.constexpr):
    pass


class ModelNew(nn.Module):
    def forward(self, x):
        return torch.conv2d(x, x)
'''

    text = critic(problem, cheating_and_broken, set()).text

    assert "S1.0" in text
    assert "F1." not in text


def test_the_findings_array_still_carries_both_families(critic, problem):
    """Suppressed in the *prompt*, not dropped from the record: the PRM trains on these."""
    review = critic(problem, DUPLICATE_ARG, set())

    assert {f["check_id"] for f in review.findings} >= {"S1.0"}
    assert all("severity" in f and "message" in f for f in review.findings)


# -- the record the loop keeps ----------------------------------------------


def test_the_review_records_whether_the_file_was_loadable(critic, problem):
    assert critic(problem, VALID, set()).data["submission_ok"] is True
    assert critic(problem, DUPLICATE_ARG, set()).data["submission_ok"] is False


def test_the_lint_counters_keep_their_meaning(critic, problem):
    """n_fail/n_warn/check_ids are the linter's, and downstream readers treat them that
    way. A submission defect must not inflate them or every historical comparison shifts."""
    review = critic(problem, DUPLICATE_ARG, set())

    assert review.data["n_fail"] == 0
    assert review.data["check_ids"] == []


def test_a_repeated_submission_defect_is_marked_as_repeated(critic, problem):
    text = critic(problem, DUPLICATE_ARG, {"S1.0"}).text

    assert "still here" in text
