"""S1.2 -- ``ModelNew`` exists but there is nothing to call.

The benchmark calls ``ModelNew(...)(inputs)``. A class with no ``forward`` -- its own or
inherited from a base defined in the same file -- has no computation to time. The linter
notices this only at *info* severity, which is never actionable, so the loop stops on it.

The inherited case is the one that needs care: `class ModelNew(Base): pass` is completely
valid when `Base.forward` is in the file, and flagging it would burn a GPU round on a
kernel that works.
"""

from __future__ import annotations

from checker.submission import SubmissionAnalyzer

PRELUDE = "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n"


def s1_2(source: str):
    findings = SubmissionAnalyzer().analyze(source, "<test>").findings
    return [f for f in findings if f.check_id == "S1.2"]


# -- fires ------------------------------------------------------------------


def test_a_ModelNew_with_no_forward_is_rejected():
    source = PRELUDE + '''

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
'''

    assert [f.check_id for f in s1_2(source)] == ["S1.2"]


def test_the_finding_is_fail_severity():
    source = PRELUDE + "\n\nclass ModelNew(nn.Module):\n    pass\n"
    assert s1_2(source)[0].severity == "fail"


def test_a_method_that_is_not_forward_does_not_count():
    source = PRELUDE + '''

class ModelNew(nn.Module):
    def run(self, x):
        return x
'''

    assert s1_2(source)


def test_a_base_defined_outside_the_file_cannot_be_credited():
    # We cannot scan torch's source, so an external base proves nothing. This is the one
    # deliberate false positive: nn.Module.forward exists but raises NotImplementedError,
    # which is exactly a zero at eval.
    source = PRELUDE + "\n\nclass ModelNew(nn.Module):\n    pass\n"
    assert s1_2(source)


# -- does not fire ----------------------------------------------------------


def test_a_ModelNew_with_its_own_forward_is_accepted():
    source = PRELUDE + '''

class ModelNew(nn.Module):
    def forward(self, x):
        return x
'''

    assert s1_2(source) == []


def test_a_forward_inherited_from_an_in_file_base_is_accepted():
    source = PRELUDE + '''

class Base(nn.Module):
    def forward(self, x):
        return x


class ModelNew(Base):
    pass
'''

    assert s1_2(source) == []


def test_a_forward_inherited_two_levels_up_is_accepted():
    source = PRELUDE + '''

class Root(nn.Module):
    def forward(self, x):
        return x


class Middle(Root):
    pass


class ModelNew(Middle):
    pass
'''

    assert s1_2(source) == []


def test_an_alias_to_a_class_with_forward_is_accepted():
    source = PRELUDE + '''

class MyKernel(nn.Module):
    def forward(self, x):
        return x


ModelNew = MyKernel
'''

    assert s1_2(source) == []


def test_a_file_with_no_ModelNew_is_left_to_s1_1():
    source = PRELUDE + "\n\nclass Model(nn.Module):\n    pass\n"

    ids = [f.check_id for f in SubmissionAnalyzer().analyze(source, "<test>").findings]

    assert "S1.2" not in ids
    assert "S1.1" in ids


def test_a_file_that_does_not_compile_is_not_also_accused_of_this():
    source = PRELUDE + "def k(x, W, W):\n    return x\n"
    assert s1_2(source) == []
