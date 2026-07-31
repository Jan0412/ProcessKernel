"""S1.3 -- a module alias is used but never bound.

``class ModelNew(nn.Module)`` in a file that never imports ``nn`` raises ``NameError`` the
moment the evaluator loads it, and the linter reports nothing: it reads the AST, where
``nn.Module`` is a perfectly well-formed attribute access.

**The predicate is deliberately narrow, and the negative controls below matter more than
the positives.** The general "any unbound attribute root" version flags 389 files in the
corpus, and its tail is `self` (19), `bn`, `init`, `W`, `B`, `F`, `x` -- names that a naive
binder misses because it does not model lambda arguments, comprehension targets or
walrus assignments. A false positive here marks a *working* kernel dirty and burns a GPU
round, which is worse than the bug being fixed. So v1 fires only on names that are never
local variables: the module aliases actually evidenced in the corpus.
"""

from __future__ import annotations

from checker.submission import SubmissionAnalyzer

FULL_PRELUDE = (
    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
    "import triton\nimport triton.language as tl\nimport math\n"
)

BODY = '''

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
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


def s1_3(source: str):
    return findings_for(source, "S1.3")


# -- fires ------------------------------------------------------------------


def test_nn_used_without_being_imported_is_rejected():
    # 29 kernels in the corpus. `class ModelNew(nn.Module)` with no `import torch.nn as nn`.
    source = "import torch\n" + BODY

    findings = s1_3(source)

    assert [f.check_id for f in findings] == ["S1.3"]
    assert "nn" in findings[0].message


def test_the_finding_is_fail_severity():
    assert s1_3("import torch\n" + BODY)[0].severity == "fail"


def test_every_allowlisted_alias_fires_when_unbound():
    for alias in ("torch", "nn", "F", "tl", "triton", "np", "numpy", "math"):
        source = (
            "import os\n\n\nclass ModelNew:\n"
            f"    def forward(self, x):\n        return {alias}.thing(x)\n"
        )
        assert s1_3(source), f"{alias} should fire when unbound"


def test_a_use_inside_forward_still_fires():
    """`F.relu(x)` fails at call time rather than import time, but both score zero and
    the message says which."""
    source = (
        "import torch\nimport torch.nn as nn\n\n\nclass ModelNew(nn.Module):\n"
        "    def forward(self, x):\n        return F.relu(x)\n"
    )

    assert s1_3(source)


def test_the_line_number_points_at_the_use():
    source = "import torch\n" + BODY
    lineno = s1_3(source)[0].data["lineno"]

    assert source.splitlines()[lineno - 1] == "class ModelNew(nn.Module):"


# -- does not fire: the binding forms ---------------------------------------


def test_a_fully_imported_file_is_accepted():
    assert s1_3(FULL_PRELUDE + BODY) == []


def test_a_plain_import_binds_the_root():
    source = "import torch\nimport numpy\n\n\nclass ModelNew:\n" \
             "    def forward(self, x):\n        return numpy.abs(x)\n"
    assert s1_3(source) == []


def test_a_dotted_import_binds_only_its_root():
    # `import torch.nn` binds `torch`, and `torch.nn.Module` works -- but bare `nn` does not.
    source = "import torch.nn\n\n\nclass ModelNew:\n" \
             "    def forward(self, x):\n        return torch.nn.functional.relu(x)\n"
    assert s1_3(source) == []


def test_a_from_import_binds_the_name():
    source = "from torch import nn\n" + BODY
    assert s1_3(source) == []


def test_a_local_assignment_binds_it():
    source = "import torch\nnn = torch.nn\n" + BODY
    assert s1_3(source) == []


def test_a_conditional_import_binds_it():
    source = (
        "import torch\ntry:\n    import numpy as np\nexcept ImportError:\n    np = None\n"
        "\n\nclass ModelNew:\n    def forward(self, x):\n        return np.abs(x)\n"
    )
    assert s1_3(source) == []


# -- does not fire: the false-positive surface ------------------------------
# These are the names the general predicate would have flagged. Each one is working code.


def test_self_is_never_flagged():
    source = FULL_PRELUDE + '''

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2.0

    def forward(self, x):
        return x * self.scale
'''
    assert s1_3(source) == []


def test_a_function_parameter_is_never_flagged():
    source = FULL_PRELUDE + '''

def helper(bn, x):
    return bn.forward(x)

''' + BODY
    assert s1_3(source) == []


def test_a_lambda_argument_is_never_flagged():
    source = FULL_PRELUDE + "\n\nf = lambda W: W.shape\n" + BODY
    assert s1_3(source) == []


def test_a_comprehension_target_is_never_flagged():
    source = FULL_PRELUDE + "\n\nshapes = [B.shape for B in []]\n" + BODY
    assert s1_3(source) == []


def test_a_walrus_binding_is_never_flagged():
    source = FULL_PRELUDE + '''

class ModelNew(nn.Module):
    def forward(self, x):
        if (init := x.numel()) > 0:
            return init
        return x
'''
    assert s1_3(source) == []


def test_an_unbound_name_outside_the_allowlist_is_not_flagged():
    """`W.foo()` with no W is a real NameError, but the corpus shows the general
    predicate misfires on exactly these short names, so v1 stays out of it."""
    source = FULL_PRELUDE + '''

class ModelNew(nn.Module):
    def forward(self, x):
        return W.matmul(x)
'''
    assert s1_3(source) == []


def test_an_attribute_on_something_else_is_not_a_root_use():
    source = FULL_PRELUDE + '''

class ModelNew(nn.Module):
    def forward(self, x):
        return x.math.something
'''
    assert s1_3(source) == []


def test_a_file_that_does_not_compile_is_not_also_accused_of_this():
    assert s1_3("import torch\ndef k(x, W, W):\n    return x\n") == []


# -- the remaining binding constructs `_bound_anywhere` must not miss --------
# Each of these is a way to bind a name that a scope-blind walker could overlook, and
# every miss is a false positive on working code.


def test_an_except_handler_name_binds_it():
    source = (
        "import os\ntry:\n    pass\nexcept Exception as math:\n    pass\n"
        "\n\nclass ModelNew:\n    def forward(self, x):\n        return math.pi\n"
    )
    assert s1_3(source) == []


def test_a_global_declaration_binds_it():
    source = (
        "import os\n\n\ndef setup():\n    global np\n    np = None\n"
        "\n\nclass ModelNew:\n    def forward(self, x):\n        return np.abs(x)\n"
    )
    assert s1_3(source) == []


def test_a_nonlocal_declaration_binds_it():
    source = (
        "import os\n\n\ndef outer():\n    tl = None\n\n    def inner():\n"
        "        nonlocal tl\n        return tl.arange(0, 4)\n    return inner\n"
        "\n\nclass ModelNew:\n    def forward(self, x):\n        return x\n"
    )
    assert s1_3(source) == []


def test_a_vararg_binds_it():
    source = (
        "import os\n\n\ndef f(*math):\n    return math\n"
        "\n\nclass ModelNew:\n    def forward(self, x):\n        return math.pi\n"
    )
    assert s1_3(source) == []


def test_a_kwarg_binds_it():
    source = (
        "import os\n\n\ndef f(**torch):\n    return torch\n"
        "\n\nclass ModelNew:\n    def forward(self, x):\n        return torch.abs(x)\n"
    )
    assert s1_3(source) == []


def test_a_keyword_only_argument_binds_it():
    source = (
        "import os\n\n\ndef f(*, nn):\n    return nn\n"
        "\n\nclass ModelNew:\n    def forward(self, x):\n        return nn.Linear(1, 1)\n"
    )
    assert s1_3(source) == []


def test_a_deleted_name_counts_as_bound():
    source = (
        "import os\nimport numpy as np\ndel np\n"
        "\n\nclass ModelNew:\n    def forward(self, x):\n        return np.abs(x)\n"
    )
    # Scope-blind and deliberately over-generous: `del` proves the name existed, and
    # erring towards silence is the whole design of this predicate.
    assert s1_3(source) == []
