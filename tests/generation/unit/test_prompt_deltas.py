"""``processkernel.generation.core.prompt_deltas``: the two blocks the PRM's training runs
appended to the prompt. Off, they must change nothing.
"""

from __future__ import annotations

import pytest

from processkernel.generation.core.model import Problem
from processkernel.generation.core.prompt_deltas import DELTA_ORDER, apply_deltas, parse_deltas

REF = (
    "import torch\n"
    "import torch.nn as nn\n\n\n"
    "class Model(nn.Module):\n"
    "    def forward(self, x):\n"
    "        return x * 2\n\n\n"
    "def get_inputs():\n"
    "    return [torch.rand([48, 48, 48, 48])]\n\n\n"
    "def get_init_inputs():\n"
    "    return [48]\n"
)
BOTH = frozenset({"contract", "precision"})

# Rendered by the code that generated the paper's level-7 runs.
PAPER_RUNS = (
    "BASE\n\n## Input contract\nModelNew is constructed as ModelNew(48) and called with 1 "
    "positional inputs, all on CUDA and contiguous:\n  arg 1: float32, shape (48, 48, 48, 48)\n"
    "This supersedes any shape stated in comments or docstrings above.\n\n"
    "## Numerical precision\nCorrectness is checked against the PyTorch reference running in "
    "true FP32, with a\ntolerance of 1e-4. Triton's tl.dot defaults to TF32 on this GPU, which "
    "carries roughly\n1e-3 of error and therefore FAILS this check. If you call tl.dot on "
    "float32 inputs you\nmust pass input_precision=\"ieee\". Note also that cuBLAS is already "
    "near-optimal for FP32\nmatmul here: fusing work around torch.matmul usually beats "
    "reimplementing the GEMM.\n"
)


def problem(ref: str = REF, pid: int = 0, level: int = 6) -> Problem:
    return Problem(level=level, problem_id=pid, name=f"{pid}_Demo.py", ref_arch_src=ref)


def test_both_blocks_render_exactly_as_in_the_paper_runs():
    assert apply_deltas("BASE\n", problem(), BOTH) == PAPER_RUNS


def test_no_blocks_leave_the_prompt_untouched():
    assert apply_deltas("BASE PROMPT\n", problem(), frozenset()) == "BASE PROMPT\n"


def test_only_the_two_paper_blocks_exist_in_a_fixed_order():
    assert DELTA_ORDER == ("contract", "precision")


def test_parse_deltas_reads_a_comma_list():
    assert parse_deltas("contract,precision") == BOTH
    assert parse_deltas(" precision , precision ") == frozenset({"precision"})
    assert parse_deltas("") == parse_deltas(None) == frozenset()


@pytest.mark.parametrize("spec", ["contract,nonsense", "pitfalls", "hardware"])
def test_parse_deltas_rejects_any_other_block(spec):
    with pytest.raises(ValueError, match="unknown prompt delta"):
        parse_deltas(spec)


def test_the_blocks_follow_delta_order_not_the_set():
    out = apply_deltas("BASE\n", problem(), frozenset({"precision", "contract"}))
    assert out.index("## Input contract") < out.index("## Numerical precision")


def test_contract_states_the_constructor_and_every_input():
    two = REF.replace("return [torch.rand([48, 48, 48, 48])]",
                      "return [torch.rand([48, 48, 48, 48]), torch.rand([48, 48, 48, 48])]")
    out = apply_deltas("BASE\n", problem(two, 19), frozenset({"contract"}))
    assert "ModelNew(48)" in out and "2 positional inputs" in out
    assert out.count("float32, shape (48, 48, 48, 48)") == 2


def test_contract_reports_the_delivered_dtype_not_the_declared_one():
    # eval casts every input to fp32, so a declared int64 arrives as float32.
    ref = (
        "import torch\n"
        "def get_inputs():\n"
        "    return [torch.rand([48, 48]), torch.ones([48], dtype=torch.int64)]\n"
        "def get_init_inputs():\n"
        "    return [48]\n"
    )
    out = apply_deltas("BASE\n", problem(ref, 92), frozenset({"contract"}))
    assert "int64" not in out and out.count("float32") == 2


@pytest.mark.parametrize(("init", "ctor"), [("    return []", "ModelNew()"),
                                            ("    return 7", "ModelNew(7)")])
def test_contract_renders_any_constructor(init, ctor):
    out = apply_deltas("BASE\n", problem(REF.replace("    return [48]", init)),
                       frozenset({"contract"}))
    assert ctor in out


def test_contract_uses_the_last_get_init_inputs():
    # Every level-6 file defines it twice; the converter's footer wins.
    ref = REF + "\n\ndef get_init_inputs():\n    return [99]\n"
    out = apply_deltas("BASE\n", problem(ref, 1), frozenset({"contract"}))
    assert "ModelNew(99)" in out and "ModelNew(48)" not in out


@pytest.mark.parametrize("ref", [
    "x = 1\n",                                          # no inputs at all
    "def (:\n",                                         # does not parse
    REF.replace("    return [48]\n", "    pass\n"),     # get_init_inputs returns nothing
    REF.replace("def get_init_inputs():\n    return [48]\n", ""),  # no get_init_inputs
    REF.replace("    return [48]", "    return [n]"),  # not a literal
    # an input sized by a module constant, as in every KernelBench level-1/2/3 file
    "import torch\nbatch = 16\ndef get_inputs():\n    return [torch.rand([48, 48]), "
    "torch.rand(batch, 4)]\ndef get_init_inputs():\n    return [48]\n",
])
def test_contract_is_left_out_rather_than_partial_or_wrong(ref):
    assert apply_deltas("BASE\n", problem(ref, 2), frozenset({"contract"})) == "BASE\n"


def test_contract_is_left_out_when_the_shape_reader_raises(monkeypatch):
    from processkernel.checker.lint import shapes

    def boom(src):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(shapes, "shapes_from_source", boom)
    assert apply_deltas("BASE\n", problem(), frozenset({"contract"})) == "BASE\n"


def test_precision_is_the_same_for_every_problem():
    a = apply_deltas("BASE\n", problem(), frozenset({"precision"}))
    assert apply_deltas("BASE\n", problem("z = 0\n", 7, 1), frozenset({"precision"})) == a
    assert 'input_precision="ieee"' in a and "1e-4" in a
