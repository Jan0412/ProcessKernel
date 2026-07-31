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


def findings_for(source: str, check_id: str):
    """Findings for one check, and proof the check did not simply crash.

    Registry.run catches a raising check into model.notes and moves on, so a broken
    predicate returns the same empty list a satisfied one does. Asserting on the notes is
    what tells those two apart."""
    report = SubmissionAnalyzer().analyze(source, "<test>")
    crashed = [n for n in report.summary.get("notes", []) if " raised " in n]
    assert not crashed, crashed
    return [f for f in report.findings if f.check_id == check_id]


def s1_2(source: str):
    return findings_for(source, "S1.2")


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


def test_an_unrelated_module_level_assignment_is_skipped():
    """The alias scan walks every top-level statement; a file full of constants must not
    confuse it for one."""
    source = PRELUDE + '''

BLOCK = 128
SCALE = 2.0


class MyKernel(nn.Module):
    def forward(self, x):
        return x


ModelNew = MyKernel
'''

    assert s1_2(source) == []


def test_an_alias_to_something_that_is_not_a_class_is_left_alone():
    # `ModelNew = make_model()` binds the name, so S1.1 is satisfied, and there is no
    # class definition to inspect for a forward. Silence is the only honest answer.
    source = PRELUDE + '''

def make_model():
    return None


ModelNew = make_model()
'''

    assert s1_2(source) == []


def test_an_alias_to_a_class_without_forward_is_rejected_through_the_alias():
    """The alias hop has to be followed in both directions. Resolving it only when the
    answer is "accepted" would let `ModelNew = Empty` through, and that scores zero."""
    source = PRELUDE + '''

class Empty(nn.Module):
    def __init__(self):
        super().__init__()


ModelNew = Empty
'''

    findings = s1_2(source)

    assert [f.check_id for f in findings] == ["S1.2"]
    assert findings[0].data["cls"] == "Empty"


def test_the_alias_must_match_the_entry_name():
    """`Other = Empty` is not the evaluator's entry point, so it says nothing about
    whether ModelNew has a forward."""
    source = PRELUDE + '''

class Empty(nn.Module):
    pass


class ModelNew(nn.Module):
    def forward(self, x):
        return x


Other = Empty
'''

    assert s1_2(source) == []
