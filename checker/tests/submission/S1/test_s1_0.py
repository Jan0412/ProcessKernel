"""S1.0 -- the file does not compile.

Every test here asserts two things: that ``ast.parse`` *succeeds*, and that S1.0 fires
anyway. That pairing is the point. The linter's front end uses ``ast.parse``, so each of
these files is reported ``parse_status="ok"``, lints clean, and then raises SyntaxError the
moment the evaluator imports it. If someone ever "simplifies" S1.0 back to ``ast.parse``,
these tests are what fails.

S1.0 is a generic gate, not a list of error types: it hands the source to CPython and asks.
"""

from __future__ import annotations

import ast

import pytest

from checker.submission import SubmissionAnalyzer

PRELUDE = "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n"

ENTRY = '''

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
'''


def findings(source: str, path: str = "<test>"):
    return SubmissionAnalyzer().analyze(source, path).findings


def s1_0(source: str):
    return [f for f in findings(source) if f.check_id == "S1.0"]


# -- the constructs the AST accepts and CPython does not --------------------
# Nine of eleven probed constructs differ between ast.parse and compile(); these are the
# ones that actually appear, or are one keyword away from appearing, in generated kernels.

DIVERGENT = {
    "duplicate_argument": "def k(x, W, W):\n    return x\n",
    "duplicate_kernel_argument": (
        "@triton.jit\n"
        "def k(x_ptr, o_ptr, stride_w, stride_w, BLOCK: tl.constexpr):\n"
        "    pass\n"
    ),
    "return_outside_function": "x = 1\nreturn x\n",
    "nonlocal_at_module_level": "def f():\n    pass\nnonlocal x\n",
    "yield_outside_function": "yield 1\n",
    "await_outside_async": "def f():\n    await g()\n",
    "break_outside_loop": "break\n",
    "starred_assignment": "*a = [1, 2]\n",
    "duplicate_keyword_argument": "f(a=1, a=2)\n",
    "global_after_parameter": "def f(x):\n    global x\n    return x\n",
}


@pytest.mark.parametrize("name", sorted(DIVERGENT))
def test_a_file_the_ast_accepts_but_python_cannot_load_is_rejected(name):
    source = PRELUDE + DIVERGENT[name] + ENTRY

    ast.parse(source)  # the linter's front end is happy with this file

    assert [f.check_id for f in s1_0(source)] == ["S1.0"]


@pytest.mark.parametrize("name", sorted(DIVERGENT))
def test_the_finding_is_fail_severity(name):
    # A loadability defect is never advice: the file scores zero until it is fixed.
    assert s1_0(PRELUDE + DIVERGENT[name] + ENTRY)[0].severity == "fail"


# -- the real corpus's dominant case ----------------------------------------


def test_the_duplicate_parameter_is_named_in_the_message():
    """181 of the 217 non-compilable kernels duplicate one parameter name. Telling the
    model *which* one is the difference between a fixable round and a wasted one."""
    source = PRELUDE + "@triton.jit\ndef k(a, W, b, W):\n    pass\n" + ENTRY

    message = s1_0(source)[0].message

    assert "W" in message
    assert "duplicate argument" in message


def test_the_line_number_is_reported_when_python_gives_one():
    source = PRELUDE + "def k(x, W, W):\n    return x\n" + ENTRY

    finding = s1_0(source)[0]

    assert finding.data["lineno"] == 5


# -- compile() raises a wider family than its name suggests -----------------


def test_a_null_byte_is_a_value_error_and_still_fires():
    # compile() raises ValueError, not SyntaxError. Narrowing the except would let the
    # most broken files through *because* they are too broken to classify.
    assert s1_0("x = 1\n\x00\n")


def test_bad_indentation_is_an_indentation_error_and_still_fires():
    assert s1_0(PRELUDE + "def f():\nreturn 1\n")


def test_a_finding_without_a_line_number_omits_the_key():
    # Only SyntaxError and its subclasses carry a lineno. inspect_trace joins findings to
    # source lines through data["lineno"], so inventing one would point it at the wrong
    # line; the key is absent instead.
    findings_ = s1_0("x = 1\n\x00\n")

    assert findings_
    assert "lineno" not in findings_[0].data


# -- the boundary: what S1.0 must NOT reject --------------------------------


def test_a_valid_kernel_is_not_rejected():
    source = PRELUDE + (
        "\n@triton.jit\n"
        "def k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
        "    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)\n"
    ) + ENTRY

    assert s1_0(source) == []


def test_a_wrong_but_loadable_kernel_is_not_rejected():
    """The gate answers "can the evaluator load and call this", never "is the answer
    right". A forward that returns x * 2 where a convolution was wanted is eval's problem;
    claiming it here would make the gate unfalsifiable."""
    source = PRELUDE + '''

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2
'''

    assert s1_0(source) == []


def test_an_empty_file_is_not_a_compile_failure():
    # Empty source compiles fine. It has no ModelNew, which is S1.1's business, not S1.0's.
    assert s1_0("") == []
