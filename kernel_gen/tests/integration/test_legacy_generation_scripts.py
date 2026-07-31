"""KGEN-15: the legacy ``generate_kernels_*``/``generate_kernelbook_samples`` scripts
used to carry their own copy of ``extract_code_block`` (``generate_kernelbook_samples``
imported ``generate_kernels_samples``'s). The copies never got the KGEN-9 empty-block
filter, so an empty fenced block still ranked as a valid candidate and won -- measured
at 38 of 41 divergences from ``core/text.py`` over the 10,510-completion trace corpus,
each an empty string shipped where the model had written a complete, loadable kernel.

Fix: the three scripts import ``extract_code_block`` from ``kernel_gen.core.text``
instead of defining it. These tests pin that: no local definition survives, all three
resolve to the exact same function object as ``core.text`` (so a future edit can't drift
one copy again), and the two real completions the golden corpus catalogued under
``kgen9_broken`` -- which the pre-fix legacy copy extracted to ``""`` -- now come back
through each script as the oracle kernel.

Imported "flat" (``kernel_gen/`` on ``sys.path``, bare module names) rather than as
``kernel_gen.generate_kernels_samples`` -- that's how these scripts are actually
launched (``python generate_kernels_samples.py`` from within ``kernel_gen/``, or
``python kernel_gen/generate_kernelbook_samples.py``), and it is load-bearing here:
the package-qualified form makes each script's own ``sys.path.insert`` resolve
``core.text`` as a second, distinct module object, which would fail the identity
check below for a reason that has nothing to do with the bug.
"""

from __future__ import annotations

import ast
import importlib
import json
import pathlib
import sys

import pytest

from kernel_gen.core.text import extract_code_block as core_extract_code_block

_KERNEL_GEN_DIR = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_WITH_OWN_COPY = ["generate_kernels_samples", "generate_kernels_reranked"]
_ALL_SCRIPTS = [*_SCRIPTS_WITH_OWN_COPY, "generate_kernelbook_samples"]

_CORPUS = pathlib.Path(__file__).parents[1] / "fixtures" / "completions" / "corpus.jsonl"
KGEN9 = [
    json.loads(line)
    for line in _CORPUS.read_text().splitlines()
    if json.loads(line)["category"] == "kgen9_broken"
]


def _load_flat(module_name: str):
    added = str(_KERNEL_GEN_DIR) not in sys.path
    if added:
        sys.path.insert(0, str(_KERNEL_GEN_DIR))
    try:
        return importlib.import_module(module_name)
    finally:
        if added:
            sys.path.remove(str(_KERNEL_GEN_DIR))


@pytest.mark.parametrize("module_name", _SCRIPTS_WITH_OWN_COPY)
def test_the_legacy_script_no_longer_defines_its_own_extractor(module_name):
    source = (_KERNEL_GEN_DIR / f"{module_name}.py").read_text()
    defined = {n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)}
    assert "extract_code_block" not in defined


@pytest.mark.parametrize("module_name", _ALL_SCRIPTS)
def test_the_legacy_scripts_extractor_is_the_core_text_one(module_name):
    core_text = _load_flat("core.text")
    mod = _load_flat(module_name)
    assert mod.extract_code_block is core_text.extract_code_block


@pytest.mark.parametrize("module_name", _ALL_SCRIPTS)
@pytest.mark.parametrize("case", KGEN9, ids=[c["id"] for c in KGEN9])
def test_the_legacy_scripts_recover_the_kgen9_empty_block_case(module_name, case):
    mod = _load_flat(module_name)
    result = mod.extract_code_block(case["raw"])
    assert result == case["oracle"]
    assert result != ""


def test_the_kgen9_corpus_still_diverges_on_core_text_too():
    # Sanity check on the fixture itself: the oracle really is what core/text.py
    # returns today, not a stale value the corpus happened to carry.
    for case in KGEN9:
        assert core_extract_code_block(case["raw"]) == case["oracle"]
