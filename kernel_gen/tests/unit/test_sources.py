"""``kernel_gen.core.sources``: dataset row -> ``Problem``, and the row-drop paths.

Zero direct coverage before. The audit cleared the index-vs-id hazard here (index picks
the row, ``problem_id`` comes from the name), so these tests pin that resolution so it
cannot silently regress, and pin the silent-drop paths (oversize / unconvertible rows)
that are the same class as the extraction bug: information lost at a seam, counted only
in a print.

The HF dataset is faked -- a list of dicts supports ``[i]``, ``len`` and ``.get`` -- so
nothing here touches the network.
"""

from __future__ import annotations

import datasets
import pytest

from kernel_gen import kernelbook_convert
from kernel_gen.core import sources


def _fake_loader(rows):
    def load_dataset(name, split=None):
        return rows

    return load_dataset


# -- KernelBench: index picks the row, id comes from the name --------------


def test_index_selects_the_row_but_the_id_comes_from_the_name(monkeypatch):
    # The hazard the audit checked: the HF split is lexicographic, so index 0 is not
    # problem 0. `spec` indexes the split; the problem_id is parsed from the name.
    rows = [
        {"name": "10_Matmul.py", "code": "SRC_A"},
        {"name": "1_Add.py", "code": "SRC_B"},
        {"name": "2_Sub.py", "code": "SRC_C"},
    ]
    monkeypatch.setattr(datasets, "load_dataset", _fake_loader(rows))

    probs = sources.load_kernelbench_problems("ds", level=1, spec="0-1")

    assert [p.problem_id for p in probs] == [10, 1]  # from the NAME, not the index
    assert [p.ref_arch_src for p in probs] == ["SRC_A", "SRC_B"]  # index picks the row
    assert all(p.level == 1 for p in probs)


def test_all_rows_takes_the_whole_split_in_order(monkeypatch):
    rows = [{"name": f"{i}_P.py", "code": str(i)} for i in range(4)]
    monkeypatch.setattr(datasets, "load_dataset", _fake_loader(rows))

    probs = sources.load_kernelbench_problems("ds", level=1, all_rows=True)
    assert [p.problem_id for p in probs] == [0, 1, 2, 3]


def test_an_out_of_range_index_is_skipped_not_fatal(monkeypatch):
    rows = [{"name": "0_P.py", "code": "x"}]
    monkeypatch.setattr(datasets, "load_dataset", _fake_loader(rows))

    probs = sources.load_kernelbench_problems("ds", level=1, spec="0,9")  # 9 is out of range
    assert [p.problem_id for p in probs] == [0]


def test_neither_spec_nor_all_is_an_error(monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", _fake_loader([{"name": "0_P.py", "code": "x"}]))
    with pytest.raises(ValueError, match="Provide"):
        sources.load_kernelbench_problems("ds", level=1)


# -- KernelBook: rows can drop, and the count must be right ----------------


def test_oversize_and_unconvertible_rows_are_dropped(monkeypatch):
    rows = [
        {"module_name": "A", "python_code": "short"},        # converts fine
        {"module_name": "B", "python_code": "x" * 30000},    # too long -> dropped
        {"module_name": "C", "python_code": "unconvertible"},  # raises -> dropped
    ]
    monkeypatch.setattr(datasets, "load_dataset", _fake_loader(rows))

    def fake_convert(code, name):
        if name == "C":
            raise kernelbook_convert.ConversionError("no module to rewrite")
        return f"REF({name})"

    monkeypatch.setattr(kernelbook_convert, "convert_row", fake_convert)

    probs = sources.load_kernelbook_problems("ds", level=5, all_rows=True, max_src_chars=24000)

    assert len(probs) == 1  # only row 0 survives
    assert probs[0].problem_id == 0  # KernelBook: the id IS the index
    assert probs[0].ref_arch_src == "REF(A)"
    assert probs[0].name == "A"
    assert probs[0].level == 5  # the pseudo-level, carried through


def test_the_row_index_is_the_problem_id_even_after_earlier_rows_drop(monkeypatch):
    # If a dropped row shifted the id, every downstream filename would point at the
    # wrong reference. The id must track the ORIGINAL index, not the surviving position.
    rows = [
        {"module_name": "A", "python_code": "x" * 30000},  # dropped
        {"module_name": "B", "python_code": "ok"},          # survives at index 1
    ]
    monkeypatch.setattr(datasets, "load_dataset", _fake_loader(rows))
    monkeypatch.setattr(kernelbook_convert, "convert_row", lambda code, name: f"REF({name})")

    probs = sources.load_kernelbook_problems("ds", level=5, all_rows=True)
    assert [p.problem_id for p in probs] == [1]  # not 0


# -- dispatch --------------------------------------------------------------


def test_load_problems_dispatches_on_dataset(monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", _fake_loader([{"name": "0_P.py", "code": "x"}]))
    probs = sources.load_problems("kernelbench", dataset_name="ds", level=1, all_rows=True)
    assert probs[0].problem_id == 0


def test_load_problems_rejects_an_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        sources.load_problems("kernelzoo", dataset_name="ds", level=1, all_rows=True)


def test_a_kernelbook_row_out_of_range_is_skipped_not_fatal(monkeypatch):
    rows = [{"module_name": "A", "python_code": "ok"}]
    monkeypatch.setattr(datasets, "load_dataset", _fake_loader(rows))
    monkeypatch.setattr(kernelbook_convert, "convert_row", lambda code, name: f"REF({name})")

    probs = sources.load_kernelbook_problems("ds", level=5, spec="0,9")  # 9 is out of range
    assert [p.problem_id for p in probs] == [0]


def test_load_problems_dispatches_to_the_kernelbook_loader(monkeypatch):
    rows = [{"module_name": "A", "python_code": "ok"}]
    monkeypatch.setattr(datasets, "load_dataset", _fake_loader(rows))
    monkeypatch.setattr(kernelbook_convert, "convert_row", lambda code, name: f"REF({name})")

    probs = sources.load_problems("kernelbook", dataset_name="ds", level=5, all_rows=True)
    assert probs[0].ref_arch_src == "REF(A)"
    assert probs[0].level == 5
