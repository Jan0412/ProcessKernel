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


# -- the staged level dir: the reference eval actually scores ---------------
#
# The bug these pin: converting a KernelBook row in-process yields an UNSCALED
# reference, while eval reads a dir staged by `convert_kernelbook.py --scale`. The
# prompt then describes a 4x4 problem that is graded at 2048x2048, and the linter's
# byte estimates -- read off this same ref_arch_src -- are wrong by the same factor.


def _stage(tmp_path, files: dict[str, str]):
    level = tmp_path / "level6"
    level.mkdir()
    for name, body in files.items():
        (level / name).write_text(body, encoding="utf-8")
    return level


def test_the_staged_file_is_the_reference_verbatim(tmp_path):
    # The whole point: bytes on disk == bytes in the prompt. No re-conversion, no
    # normalisation, nothing that could reintroduce a divergence from eval.
    src = "def get_inputs():\n    return [torch.rand([2048, 2048])]\n"
    level = _stage(tmp_path, {"10001_GCN.py": src})

    probs = sources.load_local_problems(str(level), level=6, all_rows=True)

    assert len(probs) == 1
    assert probs[0].ref_arch_src == src  # verbatim, scaled dims intact
    assert probs[0].problem_id == 10001  # from the filename prefix
    assert probs[0].name == "10001_GCN.py"
    assert probs[0].level == 6


def test_ids_come_from_the_filename_not_the_directory_order(tmp_path):
    # scandir order is arbitrary; the id is the filename prefix and --all sorts by it.
    level = _stage(tmp_path, {"10_B.py": "b", "2_C.py": "c", "1_A.py": "a"})

    probs = sources.load_local_problems(str(level), level=6, all_rows=True)

    assert [p.problem_id for p in probs] == [1, 2, 10]
    assert [p.ref_arch_src for p in probs] == ["a", "c", "b"]


def test_an_unstaged_id_is_reported_not_fatal(tmp_path, capsys):
    # Level 6 is missing ~1,090 of the ids in 0-18161 -- rows that would not convert or
    # failed the smoke test. Those gaps are a fact about the corpus (eval cannot score
    # them either), so a requested-but-absent id must be counted, never raise.
    level = _stage(tmp_path, {"0_A.py": "a", "3_B.py": "b"})

    probs = sources.load_local_problems(str(level), level=6, spec="0-4")

    assert [p.problem_id for p in probs] == [0, 3]
    assert "3 of the 5 requested ids are not staged" in capsys.readouterr().out


def test_a_shard_range_selects_by_problem_id(tmp_path):
    # The job array shards on --problems, so ranges must stay disjoint and id-keyed even
    # though the ids are sparse.
    level = _stage(tmp_path, {f"{i}_P.py": str(i) for i in (0, 1, 5, 9, 10, 11)})

    lo = sources.load_local_problems(str(level), level=6, spec="0-9")
    hi = sources.load_local_problems(str(level), level=6, spec="10-19")

    assert [p.problem_id for p in lo] == [0, 1, 5, 9]
    assert [p.problem_id for p in hi] == [10, 11]
    assert not {p.problem_id for p in lo} & {p.problem_id for p in hi}


def test_oversize_references_are_dropped_on_the_converted_source(tmp_path):
    # Under --ref-dir the budget applies to the staged file, because that string is what
    # the prompt carries.
    level = _stage(tmp_path, {"0_Small.py": "x" * 10, "1_Huge.py": "x" * 30000})

    probs = sources.load_local_problems(str(level), level=6, all_rows=True, max_src_chars=24000)

    assert [p.problem_id for p in probs] == [0]


def test_non_reference_files_in_the_level_dir_are_ignored(tmp_path):
    # A staged dir also holds manifest.json / conversion_stats.json, and level6 has a
    # stray level5.zip. None of them is a problem.
    level = _stage(
        tmp_path,
        {"0_A.py": "a", "manifest.json": "{}", "conversion_stats.json": "{}", "level5.zip": "PK"},
    )

    probs = sources.load_local_problems(str(level), level=6, all_rows=True)
    assert [p.problem_id for p in probs] == [0]


def test_a_subdirectory_is_not_mistaken_for_a_reference(tmp_path):
    level = _stage(tmp_path, {"0_A.py": "a"})
    (level / "9_NotAFile.py").mkdir()

    probs = sources.load_local_problems(str(level), level=6, all_rows=True)
    assert [p.problem_id for p in probs] == [0]


def test_duplicate_ids_raise_rather_than_silently_picking_one(tmp_path):
    # eval's find_ref returns whatever the glob yields first. If generation picked the
    # other file, the prompt and the reference would diverge again -- the exact bug this
    # loader exists to make impossible -- so stop instead.
    level = _stage(tmp_path, {"7_A.py": "a", "7_B.py": "b"})

    with pytest.raises(ValueError, match="two references for problem 7"):
        sources.load_local_problems(str(level), level=6, all_rows=True)


def test_a_missing_or_empty_ref_dir_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="is not a directory"):
        sources.load_local_problems(str(tmp_path / "nope"), level=6, all_rows=True)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no <id>_<Name>.py"):
        sources.load_local_problems(str(empty), level=6, all_rows=True)


def test_neither_spec_nor_all_is_an_error_for_the_local_loader(tmp_path):
    level = _stage(tmp_path, {"0_A.py": "a"})
    with pytest.raises(ValueError, match="Provide"):
        sources.load_local_problems(str(level), level=6)


def test_ref_dir_overrides_the_dataset_loader_without_touching_hf(tmp_path, monkeypatch):
    # --ref-dir must not fall back to the network, and must not need --dataset-name.
    def explode(*a, **k):
        raise AssertionError("load_dataset must not be called under --ref-dir")

    monkeypatch.setattr(datasets, "load_dataset", explode)
    level = _stage(tmp_path, {"4_A.py": "SCALED"})

    probs = sources.load_problems(
        "kernelbook", ref_dir=str(level), dataset_name="ds", level=6, all_rows=True
    )

    assert [p.problem_id for p in probs] == [4]
    assert probs[0].ref_arch_src == "SCALED"


def test_ref_dir_still_validates_the_dataset_name(tmp_path):
    # --dataset keeps deciding level vs pseudo_level in the written config, so a typo
    # must still fail even though it no longer selects a loader.
    level = _stage(tmp_path, {"0_A.py": "a"})
    with pytest.raises(ValueError, match="unknown dataset"):
        sources.load_problems("kernelzoo", ref_dir=str(level), level=6, all_rows=True)


def test_ref_dir_works_for_kernelbench_levels_too(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "load_dataset", _fake_loader([]))
    level = _stage(tmp_path, {"19_ReLU.py": "ref"})

    probs = sources.load_problems(
        "kernelbench", ref_dir=str(level), dataset_name="ds", level=1, all_rows=True
    )

    assert probs[0].problem_id == 19
    assert probs[0].level == 1


# -- sharding an array run over a sparse id space --------------------------


def _sharded_ids(level, nshards):
    """Every problem id each shard would actually load, via the real loader."""
    return [
        [
            p.problem_id
            for p in sources.load_local_problems(
                str(level), level=6, spec=sources.shard_range(str(level), k, nshards)
            )
        ]
        for k in range(nshards)
    ]


def test_shards_partition_every_problem_exactly_once(tmp_path):
    # The property the array run depends on: disjoint (no slot generated twice, no
    # corrupted merge) and complete (no problem silently dropped).
    ids = [0, 1, 2, 7, 8, 40, 41, 42, 99, 100, 500, 5000]
    level = _stage(tmp_path, {f"{i}_P.py": str(i) for i in ids})

    for nshards in (1, 2, 3, 5, 12):
        shards = _sharded_ids(level, nshards)
        flat = [pid for shard in shards for pid in shard]
        assert sorted(flat) == ids, f"nshards={nshards} did not cover every id"
        assert len(flat) == len(set(flat)), f"nshards={nshards} produced an overlap"


def test_shards_are_balanced_despite_a_lopsided_id_space(tmp_path):
    # Cutting the id RANGE would put 9 of these 10 problems in the last shard; cutting
    # the sorted id LIST splits them 5/5. This is the whole reason for the helper.
    ids = [0, 1, 2, 3, 4, 9995, 9996, 9997, 9998, 9999]
    level = _stage(tmp_path, {f"{i}_P.py": str(i) for i in ids})

    shards = _sharded_ids(level, 2)
    assert [len(s) for s in shards] == [5, 5]


def test_a_shard_range_never_reaches_into_the_next_shard(tmp_path):
    # The ranges are widened to span their chunk, so pin that the widening stops short
    # of the next chunk's first id.
    ids = [0, 5, 100, 101, 900, 1000]
    level = _stage(tmp_path, {f"{i}_P.py": str(i) for i in ids})

    ranges = [sources.shard_range(str(level), k, 3) for k in range(3)]
    assert ranges == ["0-5", "100-101", "900-1000"]


def test_shard_bounds_are_validated(tmp_path):
    level = _stage(tmp_path, {"0_A.py": "a", "1_B.py": "b"})

    with pytest.raises(ValueError, match="out of range"):
        sources.shard_range(str(level), 2, 2)
    with pytest.raises(ValueError, match="out of range"):
        sources.shard_range(str(level), -1, 2)
    with pytest.raises(ValueError, match="nshards must be"):
        sources.shard_range(str(level), 0, 0)
    # More shards than problems would hand some task an empty chunk and crash on
    # chunk[0]; refuse up front with a message that says what to do.
    with pytest.raises(ValueError, match="3 shards over only 2 problems"):
        sources.shard_range(str(level), 0, 3)
