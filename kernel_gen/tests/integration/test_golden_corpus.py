"""Extraction against a frozen corpus of REAL Qwen3.6 completions.

Synthetic fixtures encode what I imagined the model does. This corpus is what it
actually did -- 20 completions pulled from the level 1+2 trace runs, each labelled by
behaviour and carrying its oracle (the kernel a correct extractor should return,
computed independently of ``extract_code_block``). It is the regression bedrock: a
parser change that breaks on real output fails here, and the KGEN-2 cases are real proof
the bug bites on real data, not just on a hand-built repro.

Categories (see fixtures/completions/corpus.jsonl):
  single_block / revision / revision_fragment_first  -> extraction is correct today
  unterminated_recovered  -> an unterminated final block the fallback path DID recover
  kgen2_broken            -> an unterminated final block; recovered since the KGEN-2 fix
  kgen3_broken            -> a stray closing fence hid the real block; recovered since KGEN-3
  kgen9_broken            -> unfenced answer + the sampler's trailing ```python; recovered
                             since KGEN-9. Both were catalogued as model_failed until the
                             fix showed they contain a complete ModelNew.
  kgen11_broken           -> the answer written OUTSIDE every fenced block; unreachable
                             until the ladder took outside regions as candidates
  kgen19_broken           -> a block spelling ModelNew that does not parse, which used to
                             lose to a parseable fragment with no entry class
  model_failed            -> no valid ModelNew anywhere (the model failed, not a bug)
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from kernel_gen.core.text import extract_code_block

_CORPUS = pathlib.Path(__file__).parents[1] / "fixtures" / "completions" / "corpus.jsonl"
CORPUS = [json.loads(line) for line in _CORPUS.read_text().splitlines()]
_CORRECT = ("single_block", "revision", "revision_fragment_first", "unterminated_recovered")
WELLFORMED = [c for c in CORPUS if c["category"] in _CORRECT]
KGEN2 = [c for c in CORPUS if c["category"] == "kgen2_broken"]
KGEN3 = [c for c in CORPUS if c["category"] == "kgen3_broken"]
KGEN9 = [c for c in CORPUS if c["category"] == "kgen9_broken"]
KGEN11 = [c for c in CORPUS if c["category"] == "kgen11_broken"]
KGEN19 = [c for c in CORPUS if c["category"] == "kgen19_broken"]
FAILED = [c for c in CORPUS if c["category"] == "model_failed"]


def _id(cases):
    return [c["id"] for c in cases]


def test_the_corpus_covers_every_behaviour():
    # If a category empties out (e.g. a rebuild that lost the KGEN-2 cases), the suite
    # would silently stop testing it. Fail loudly instead.
    cats = {c["category"] for c in CORPUS}
    assert {"single_block", "revision", "kgen2_broken", "kgen3_broken", "kgen9_broken",
            "kgen11_broken", "kgen19_broken", "model_failed"} <= cats


@pytest.mark.parametrize("case", WELLFORMED, ids=_id(WELLFORMED))
def test_extraction_matches_the_oracle_on_real_wellformed_completions(case):
    out = extract_code_block(case["raw"])
    assert out.strip() == case["oracle"].strip()
    assert "class ModelNew" in out
    ast.parse(out)  # a correct extraction is always valid Python


@pytest.mark.parametrize("case", FAILED, ids=_id(FAILED))
def test_a_genuinely_failed_completion_does_not_crash_extraction(case):
    # The model produced no ModelNew; the extractor returns its best effort without
    # raising, and we do not pretend a submission exists.
    out = extract_code_block(case["raw"])  # must not raise
    assert not case["oracle_has_modelnew"]
    # …and the raw really has no submission, so this stays a statement about the model.
    # Two cases sat in this category until KGEN-9 showed they contained a complete
    # ModelNew and the extractor was returning "" for them.
    assert "class ModelNew" not in case["raw"]
    assert "class ModelNew" not in out


@pytest.mark.parametrize("case", KGEN9, ids=_id(KGEN9))
def test_kgen9_real_completions_recover_an_unfenced_answer(case):
    # KGEN-9, now fixed. The model wrote its whole answer as unfenced prose-with-code and
    # the two-pass sampler then appended its ```python with nothing after it. That empty
    # block parsed (ast.parse("") succeeds), so it won the ranking and "" was shipped --
    # for round 0 that meant an EMPTY baseline kernel for a generation that succeeded.
    out = extract_code_block(case["raw"])
    assert out.strip() == case["oracle"].strip()
    assert "class ModelNew" in out
    tree = ast.parse(out)
    model_new = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelNew")
    # Instantiable AND callable -- a salvage that lost forward() would still score zero.
    assert "forward" in [m.name for m in model_new.body if isinstance(m, ast.FunctionDef)]


@pytest.mark.parametrize("case", KGEN2, ids=_id(KGEN2))
def test_kgen2_real_completions_recover_the_unterminated_final(case):
    # KGEN-2, now fixed: each of these real completions has a complete earlier block that
    # USED to win over the cut-off final ModelNew. Extraction now recovers the tail. See
    # checker/tests/KERNEL_GEN_BUGS.md.
    assert extract_code_block(case["raw"]).strip() == case["oracle"].strip()


@pytest.mark.parametrize("case", KGEN3, ids=_id(KGEN3))
def test_kgen3_real_completions_survive_a_stray_closing_fence(case):
    # KGEN-3, now fixed. These are model SUCCESSES: each raw contains a complete,
    # parseable ModelNew (the oracle) that a stray closing fence used to hide -- the old
    # regex paired that stray ``` with the real block's ```python opener and shipped the
    # inter-block prose. _fenced_blocks now ignores a bare ``` outside a block, so the
    # real block is recovered. See checker/tests/KERNEL_GEN_BUGS.md.
    out = extract_code_block(case["raw"])
    assert "class ModelNew" in out
    assert out.strip() == case["oracle"].strip()


@pytest.mark.parametrize("case", KGEN11, ids=_id(KGEN11))
def test_kgen11_real_completions_recover_an_answer_written_outside_the_fences(case):
    # KGEN-11, now fixed. The model closed a draft's fence and wrote its real answer as
    # plain text, so `extract_code_block` -- which returned as soon as any fenced block
    # existed -- shipped whatever scrap happened to be fenced. One of these shipped 95
    # characters while a 10,373-character loadable kernel sat in the same string.
    out = extract_code_block(case["raw"])
    assert out.strip() == case["oracle"].strip()
    tree = ast.parse(out)  # a correct extraction is valid Python
    model_new = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ModelNew"
    )
    # Instantiable AND callable -- a recovery that lost forward() would still score zero.
    assert "forward" in [m.name for m in model_new.body if isinstance(m, ast.FunctionDef)]
    compile(out, "<kernel>", "exec")  # and loadable, which is what eval needs


@pytest.mark.parametrize("case", KGEN19, ids=_id(KGEN19))
def test_kgen19_real_completions_keep_the_entry_class_over_a_parseable_fragment(case):
    # KGEN-19, now fixed. With no block both parsing AND defining ModelNew, the ladder
    # fell past the blocks that merely spell it and shipped a fragment with no entry class
    # -- pointing the repair round at the wrong file. None of these becomes loadable; the
    # value is that the feedback and the PRM label name the model's real kernel.
    out = extract_code_block(case["raw"])
    assert out.strip() == case["oracle"].strip()
    assert "class ModelNew" in out
