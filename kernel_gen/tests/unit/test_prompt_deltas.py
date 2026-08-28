"""``kernel_gen.core.prompt_deltas``: additive, individually-switchable prompt blocks.

The invariant the A/B rests on is that an empty delta set changes nothing -- that is
what lets an existing run stand in as the control arm.
"""

from __future__ import annotations

import pytest

from kernel_gen.core.model import Problem
from kernel_gen.core.prompt_deltas import (
    DELTA_ORDER,
    apply_deltas,
    parse_deltas,
)

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


def _problem() -> Problem:
    return Problem(level=6, problem_id=0, name="0_Demo.py", ref_arch_src=REF)


def test_empty_delta_set_leaves_the_prompt_untouched():
    prompt = "BASE PROMPT\n"
    assert apply_deltas(prompt, _problem(), frozenset()) == prompt


def test_parse_deltas_accepts_a_comma_string():
    assert parse_deltas("precision,hardware") == frozenset({"precision", "hardware"})


def test_parse_deltas_of_empty_string_is_empty():
    assert parse_deltas("") == frozenset()
    assert parse_deltas(None) == frozenset()


def test_parse_deltas_tolerates_whitespace_and_duplicates():
    assert parse_deltas(" precision , precision ,hardware") == frozenset(
        {"precision", "hardware"}
    )


def test_parse_deltas_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown prompt delta"):
        parse_deltas("contract,nonsense")


def test_delta_order_is_a_tuple_not_a_set():
    # Set iteration order varies per process; the rendered prompt must not.
    assert isinstance(DELTA_ORDER, tuple)


def test_apply_deltas_is_deterministic_across_calls():
    p = _problem()
    both = frozenset({"precision", "pitfalls"})
    assert apply_deltas("BASE\n", p, both) == apply_deltas("BASE\n", p, both)


def test_contract_block_states_constructor_args_and_every_input():
    ref = REF.replace("return [torch.rand([48, 48, 48, 48])]",
                      "return [torch.rand([48, 48, 48, 48]), torch.rand([48, 48, 48, 48])]")
    p = Problem(level=6, problem_id=19, name="19_Demo.py", ref_arch_src=ref)
    out = apply_deltas("BASE\n", p, frozenset({"contract"}))
    assert "## Input contract" in out
    assert "ModelNew(48)" in out
    assert "2 positional inputs" in out
    assert out.count("float32, shape (48, 48, 48, 48)") == 2
    assert "supersedes any shape stated in comments or docstrings" in out


def test_contract_block_reports_the_delivered_dtype_not_the_declared_one():
    # eval casts every input tensor to the eval precision, so a declared int64 tensor
    # arrives as float32. Stating int64 would contradict what the model receives.
    ref = (
        "import torch\n"
        "import torch.nn as nn\n\n\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x, labels):\n"
        "        return x\n\n\n"
        "def get_inputs():\n"
        "    return [torch.rand([48, 48]), torch.ones([48], dtype=torch.int64)]\n\n\n"
        "def get_init_inputs():\n"
        "    return [48]\n"
    )
    p = Problem(level=6, problem_id=92, name="92_Demo.py", ref_arch_src=ref)
    out = apply_deltas("BASE\n", p, frozenset({"contract"}))
    assert "int64" not in out
    assert out.count("float32") == 2


def test_contract_block_renders_a_no_argument_constructor():
    ref = REF.replace("    return [48]", "    return []")
    p = Problem(level=6, problem_id=0, name="0_Demo.py", ref_arch_src=ref)
    out = apply_deltas("BASE\n", p, frozenset({"contract"}))
    assert "ModelNew()" in out


def test_contract_block_uses_the_last_get_init_inputs_definition():
    # Every level-6 file defines get_init_inputs twice; the converter footer wins.
    ref = REF + "\n\ndef get_init_inputs():\n    return [99]\n"
    p = Problem(level=6, problem_id=1, name="1_Demo.py", ref_arch_src=ref)
    out = apply_deltas("BASE\n", p, frozenset({"contract"}))
    assert "ModelNew(99)" in out
    assert "ModelNew(48)" not in out


def test_contract_block_is_omitted_when_shapes_cannot_be_resolved():
    p = Problem(level=6, problem_id=2, name="2_Demo.py", ref_arch_src="x = 1\n")
    out = apply_deltas("BASE\n", p, frozenset({"contract"}))
    assert out == "BASE\n"


def test_contract_block_never_raises_on_unparseable_source():
    p = Problem(level=6, problem_id=3, name="3_Demo.py", ref_arch_src="def (:\n")
    assert apply_deltas("BASE\n", p, frozenset({"contract"})) == "BASE\n"


def test_contract_block_is_omitted_when_only_some_shapes_resolve():
    # shapes_from_source returns a None entry per input it cannot resolve. Every
    # official KernelBench level-1/2/3 file hits this, because they size inputs from
    # module-level constants. A partial contract would be worse than none.
    ref = (
        "import torch\n"
        "batch = 16\n"
        "def get_inputs():\n"
        "    return [torch.rand([48, 48]), torch.rand(batch, 4)]\n"
        "def get_init_inputs():\n"
        "    return [48]\n"
    )
    p = Problem(level=1, problem_id=5, name="5_Demo.py", ref_arch_src=ref)
    assert apply_deltas("BASE\n", p, frozenset({"contract"})) == "BASE\n"


def test_precision_block_names_the_tolerance_and_the_tf32_default():
    out = apply_deltas("BASE\n", _problem(), frozenset({"precision"}))
    assert "## Numerical precision" in out
    assert "1e-4" in out
    assert 'input_precision="ieee"' in out
    assert "TF32" in out


def test_pitfalls_block_carries_every_rule():
    out = apply_deltas("BASE\n", _problem(), frozenset({"pitfalls"}))
    assert "## Triton pitfalls" in out
    # One needle per rule, each matching the rule's own words -- "mask" alone would
    # also match "unmasked" and "import" does not match the capitalised bullet.
    for needle in (
        "Define ModelNew",
        "Mask every tl.load",
        "tl.constexpr",
        "@triton.jit",
        "Import everything you use",
        "Triton 3.6",
    ):
        assert needle in out


def test_static_blocks_ignore_the_problem():
    a = apply_deltas("BASE\n", _problem(), frozenset({"precision", "pitfalls"}))
    other = Problem(level=1, problem_id=7, name="7_X.py", ref_arch_src="z = 0\n")
    assert apply_deltas("BASE\n", other, frozenset({"precision", "pitfalls"})) == a


def test_blocks_are_ordered_by_delta_order_not_by_set_iteration():
    out = apply_deltas("BASE\n", _problem(), frozenset({"pitfalls", "precision"}))
    assert out.index("## Numerical precision") < out.index("## Triton pitfalls")


def test_hardware_appends_no_text_even_though_blocks_now_render():
    # `hardware` is a flag on the KernelBench call, not appended text; with real blocks
    # in place this now actually exercises the FLAG_DELTAS gate.
    assert apply_deltas("BASE\n", _problem(), frozenset({"hardware"})) == "BASE\n"
