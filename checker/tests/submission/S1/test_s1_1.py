"""S1.1 -- the file binds no ``ModelNew``.

The evaluator instantiates ``ModelNew`` by name. A file that defines a perfectly good
Triton kernel and calls its class ``Model`` scores zero, and the linter is happy with it --
``ENTRY_CLASSES`` accepts both names, which is right when linting a *reference* and wrong
when grading a *submission*. That distinction is what this check exists for.
"""

from __future__ import annotations

from checker.submission import SubmissionAnalyzer

PRELUDE = "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n"

KERNEL = '''

@triton.jit
def k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)
'''


def findings_for(source: str, check_id: str):
    """Findings for one check, and proof the check did not simply crash.

    Registry.run catches a raising check into model.notes and moves on, so a broken
    predicate returns the same empty list a satisfied one does. Asserting on the notes is
    what tells those two apart."""
    report = SubmissionAnalyzer().analyze(source, "<test>")
    crashed = [n for n in report.summary.get("notes", []) if " raised " in n]
    assert not crashed, crashed
    return [f for f in report.findings if f.check_id == check_id]


def s1_1(source: str):
    return findings_for(source, "S1.1")


# -- fires ------------------------------------------------------------------


def test_a_file_with_no_class_at_all_is_rejected():
    # Imports plus a bare @triton.jit function: nothing for the evaluator to instantiate.
    assert [f.check_id for f in s1_1(PRELUDE + KERNEL)] == ["S1.1"]


def test_the_finding_is_fail_severity():
    assert s1_1(PRELUDE + KERNEL)[0].severity == "fail"


def test_a_class_named_Model_is_rejected_and_the_message_says_so():
    """The most recoverable version of this defect: the work is done, the name is wrong."""
    source = PRELUDE + KERNEL + '''

class Model(nn.Module):
    def forward(self, x):
        return x
'''

    message = s1_1(source)[0].message

    assert "Model" in message and "ModelNew" in message


def test_a_class_with_some_other_name_is_rejected():
    source = PRELUDE + KERNEL + '''

class TritonModel(nn.Module):
    def forward(self, x):
        return x
'''

    assert s1_1(source)


def test_a_nested_ModelNew_does_not_count():
    # Defined inside another class, so `ModelNew` is not bound at module level and the
    # evaluator's lookup fails.
    source = PRELUDE + KERNEL + '''

class Wrapper:
    class ModelNew(nn.Module):
        def forward(self, x):
            return x
'''

    assert s1_1(source)


# -- does not fire ----------------------------------------------------------


def test_a_plain_ModelNew_is_accepted():
    source = PRELUDE + KERNEL + '''

class ModelNew(nn.Module):
    def forward(self, x):
        return x
'''

    assert s1_1(source) == []


def test_an_alias_binding_is_accepted():
    """`ModelNew = MyKernel` is a real pattern and it works at eval: the name is bound.
    A false positive here would mark a good kernel dirty and burn a GPU round."""
    source = PRELUDE + KERNEL + '''

class MyKernel(nn.Module):
    def forward(self, x):
        return x


ModelNew = MyKernel
'''

    assert s1_1(source) == []


def test_a_multiple_assignment_alias_is_accepted():
    source = PRELUDE + KERNEL + '''

class MyKernel(nn.Module):
    def forward(self, x):
        return x


Model = ModelNew = MyKernel
'''

    assert s1_1(source) == []


def test_an_annotated_alias_is_accepted():
    source = PRELUDE + KERNEL + '''

class MyKernel(nn.Module):
    def forward(self, x):
        return x


ModelNew: type = MyKernel
'''

    assert s1_1(source) == []


def test_an_imported_ModelNew_is_accepted():
    source = PRELUDE + "from somewhere import ModelNew\n" + KERNEL

    assert s1_1(source) == []


def test_a_file_that_does_not_compile_is_not_also_accused_of_this():
    """S1.0 reports the cause; piling on a consequence would send the model chasing a
    missing class that is actually right there, three lines below a syntax error."""
    source = PRELUDE + "def k(x, W, W):\n    return x\n"

    findings = SubmissionAnalyzer().analyze(source, "<test>").findings

    assert [f.check_id for f in findings] == ["S1.0"]


# -- the binding forms `_bound_by` has to understand -------------------------


def test_a_tuple_unpacking_binds_it():
    source = PRELUDE + KERNEL + '''

class MyKernel(nn.Module):
    def forward(self, x):
        return x


ModelNew, Other = MyKernel, MyKernel
'''

    assert s1_1(source) == []


def test_a_starred_unpacking_binds_it():
    source = PRELUDE + KERNEL + '''

class MyKernel(nn.Module):
    def forward(self, x):
        return x


[ModelNew, *rest] = [MyKernel]
'''

    assert s1_1(source) == []


def test_an_attribute_target_does_not_bind_it():
    """`obj.ModelNew = X` binds an attribute on obj, not the module-level name the
    evaluator looks up."""
    source = PRELUDE + KERNEL + '''

class Holder:
    pass


Holder.ModelNew = Holder
'''

    assert s1_1(source)


def test_an_augmented_assignment_counts_as_a_binding():
    source = PRELUDE + KERNEL + '''

ModelNew = None
ModelNew += 1
'''

    assert s1_1(source) == []
