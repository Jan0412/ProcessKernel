"""``prm.corpus``: what the walk finds, what it reports missing, and what the join produces."""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import pytest

from reranker.src.config import PROJECT_ROOT
from reranker.src.prm.corpus import NO_ATTEMPTS, NO_EVAL, NO_EVAL_ENTRY, iter_labeled, units
from reranker.src.prm.targets import GRADED, MEAN, MIN, target_for
from reranker.tests.prm.runfixture import attempt, one_unit, verdict, write_round

ONE = ([attempt(1, 0)], {"1": [verdict(0)]})


def labeled(unit):
    """Every row of a unit, with the drop ledger the stream filled in."""
    counts: Counter = Counter()
    return list(iter_labeled(unit, counts)), counts


# --- units(): the walk ----------------------------------------------------------------


def test_a_round_with_both_files_is_a_unit(tmp_path):
    run_dir = tmp_path / "myrun"
    write_round(run_dir, "shard_03", 2, *ONE)
    found, skipped = units([str(run_dir)], [2])
    assert skipped == []
    (unit,) = found
    assert (unit.run_name, unit.shard, unit.round) == ("myrun", "shard_03", 2)
    assert unit.attempts_path.endswith("shard_03/traces/round_2/attempts.jsonl")
    assert unit.eval_path.endswith("shard_03/rounds/round_2/eval_results.json")


def test_a_round_generated_but_not_yet_evaluated_is_skipped_and_counted(tmp_path):
    # Evaluation lags generation by design, so this is the normal state of a live corpus.
    run_dir = tmp_path / "myrun"
    write_round(run_dir, "shard_00", 0, *ONE)
    write_round(run_dir, "shard_00", 1, [attempt(1, 0)], None)
    found, skipped = units([str(run_dir)], [0, 1])
    assert [u.round for u in found] == [0]
    assert skipped == [("myrun", "shard_00", 1, NO_EVAL)]


def test_a_round_that_was_never_generated_is_skipped_as_no_attempts(tmp_path):
    run_dir = tmp_path / "myrun"
    write_round(run_dir, "shard_00", 0, *ONE)
    write_round(run_dir, "shard_00", 2, None, None)
    found, skipped = units([str(run_dir)], [0, 2])
    assert [u.round for u in found] == [0]
    assert skipped == [("myrun", "shard_00", 2, NO_ATTEMPTS)]


def test_an_evaluated_round_with_no_attempts_file_is_still_no_attempts(tmp_path):
    # Nothing to cut, so the eval file alone is not a unit -- and the reason must say why.
    run_dir = tmp_path / "myrun"
    write_round(run_dir, "shard_00", 0, None, {"1": [verdict(0)]})
    found, skipped = units([str(run_dir)], [0])
    assert found == []
    assert skipped == [("myrun", "shard_00", 0, NO_ATTEMPTS)]


def test_only_the_configured_rounds_are_walked(tmp_path):
    run_dir = tmp_path / "myrun"
    for rnd in (0, 1, 2):
        write_round(run_dir, "shard_00", rnd, *ONE)
    found, skipped = units([str(run_dir)], [1])
    assert [u.round for u in found] == [1]
    assert skipped == []


def test_shards_are_walked_in_sorted_order(tmp_path):
    run_dir = tmp_path / "myrun"
    for shard in ("shard_10", "shard_02", "shard_01"):
        write_round(run_dir, shard, 0, *ONE)
    found, _ = units([str(run_dir)], [0])
    assert [u.shard for u in found] == ["shard_01", "shard_02", "shard_10"]


def test_two_runs_that_both_contain_shard_00_give_two_distinct_units(tmp_path):
    # PLAN §9: the part file is named from (run, shard, round), so the run has to be here.
    for run in ("runA", "runB"):
        write_round(tmp_path / run, "shard_00", 0, *ONE)
    found, _ = units([str(tmp_path / "runA"), str(tmp_path / "runB")], [0])
    assert [(u.run_name, u.shard, u.round) for u in found] == [
        ("runA", "shard_00", 0),
        ("runB", "shard_00", 0),
    ]


@pytest.mark.parametrize(("cap", "expected"), [(None, 3), (1, 1), (2, 2), (99, 3)])
def test_max_shards_caps_each_run_from_the_front(tmp_path, cap, expected):
    run_dir = tmp_path / "myrun"
    for shard in ("shard_00", "shard_01", "shard_02"):
        write_round(run_dir, shard, 0, *ONE)
    found, _ = units([str(run_dir)], [0], max_shards=cap)
    assert [u.shard for u in found] == ["shard_00", "shard_01", "shard_02"][:expected]


def test_a_run_dir_that_does_not_exist_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nosuchrun"):
        units([str(tmp_path / "nosuchrun")], [0])


def test_a_run_dir_with_no_shards_raises_rather_than_building_nothing(tmp_path):
    # A path typo would otherwise produce an empty build with no error anywhere.
    (tmp_path / "myrun" / "traces").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no shard_"):
        units([str(tmp_path / "myrun")], [0])


def test_a_trailing_slash_does_not_change_the_run_name(tmp_path):
    write_round(tmp_path / "myrun", "shard_00", 0, *ONE)
    found, _ = units([str(tmp_path / "myrun") + "/"], [0])
    assert found[0].run_name == "myrun"


def test_two_run_dirs_with_the_same_basename_raise(tmp_path):
    # Everything downstream is keyed on run_name -- part files, the manifest, the skip
    # list -- so one run would overwrite the other and the loss would show up nowhere.
    for parent in ("a", "b"):
        write_round(tmp_path / parent / "v5_traced", "shard_00", 0, *ONE)
    dirs = [str(tmp_path / p / "v5_traced") for p in ("a", "b")]
    with pytest.raises(ValueError, match="collide on basename"):
        units(dirs, [0])


def test_the_same_run_dir_listed_twice_raises(tmp_path):
    write_round(tmp_path / "myrun", "shard_00", 0, *ONE)
    with pytest.raises(ValueError, match="collide on basename"):
        units([str(tmp_path / "myrun")] * 2, [0])


def test_the_same_round_listed_twice_raises(tmp_path):
    # One (run, shard, round) is one part file, so a repeat is two workers writing it at
    # once -- a torn file, and every attempt of that round in the dataset twice.
    write_round(tmp_path / "myrun", "shard_00", 0, *ONE)
    with pytest.raises(ValueError, match="rounds repeat"):
        units([str(tmp_path / "myrun")], [0, 0])


def test_a_relative_run_dir_resolves_against_the_project_not_the_cwd(tmp_path, monkeypatch):
    # config._resolve is what every other reranker path uses; os.path.abspath would make
    # the same YAML find a different tree depending on the submit dir.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=re.escape(os.path.join(PROJECT_ROOT, "runs"))):
        units(["runs/nosuchrun"], [0])


# --- iter_labeled(): the join ---------------------------------------------------------


def test_an_attempt_is_joined_to_its_own_verdict_not_to_a_neighbour(tmp_path):
    attempts = [attempt(7, 0), attempt(7, 1), attempt(9, 0)]
    verdicts = {
        "9": [verdict(0, correctness=False, runtime=9.0)],
        # out of order on both keys: the join is by id, never by position
        "7": [verdict(1, runtime=2.0), verdict(0, runtime=4.0)],
    }
    rows, counts = labeled(one_unit(tmp_path, attempts, verdicts))
    assert [(r.problem_id, r.sample_id, r.runtimes[MEAN]) for r in rows] == [
        (7, 0, 4.0),
        (7, 1, 2.0),
        (9, 0, 9.0),
    ]
    assert [r.correct for r in rows] == [True, True, False]
    assert counts == Counter()


def test_an_attempt_with_no_eval_entry_is_counted_not_yielded(tmp_path):
    attempts = [attempt(1, 0), attempt(1, 1), attempt(2, 0)]
    rows, counts = labeled(one_unit(tmp_path, attempts, {"1": [verdict(0)]}))
    assert [(r.problem_id, r.sample_id) for r in rows] == [(1, 0)]
    assert counts == Counter({NO_EVAL_ENTRY: 2})


def test_an_eval_entry_with_no_attempt_is_simply_absent(tmp_path):
    verdicts = {"1": [verdict(0), verdict(1)]}
    rows, counts = labeled(one_unit(tmp_path, [attempt(1, 0)], verdicts))
    assert [r.sample_id for r in rows] == [0]
    assert counts == Counter()


def test_the_level_comes_from_the_record_not_from_config(tmp_path):
    unit = one_unit(tmp_path, [attempt(1, 0, level=3)], {"1": [verdict(0)]})
    rows, _ = labeled(unit)
    assert rows[0].level == 3
    assert rows[0].stem == "level_3_problem_1_sample_0_kernel"


def test_each_runtime_is_read_from_its_own_key(tmp_path):
    unit = one_unit(tmp_path, [attempt(1, 0)], {"1": [verdict(0, runtime=8.0, min_runtime=2.0)]})
    rows, _ = labeled(unit)
    assert rows[0].runtimes == {MEAN: 8.0, MIN: 2.0}


def test_a_verdict_that_was_never_timed_yields_none_on_both_stats(tmp_path):
    # The 2 correct kernels carrying `runtime: -1.0` with an empty runtime_stats reach
    # here; -1.0 is passed through and rejected by targets, the absent min is None.
    entry = verdict(0, runtime=-1.0, min_runtime=None)
    rows, _ = labeled(one_unit(tmp_path, [attempt(1, 0)], {"1": [entry]}))
    assert rows[0].runtimes == {MEAN: -1.0, MIN: None}


def test_integer_runtimes_from_json_become_floats(tmp_path):
    unit = one_unit(tmp_path, [attempt(1, 0)], {"1": [verdict(0, runtime=3, min_runtime=2)]})
    rows, _ = labeled(unit)
    assert rows[0].runtimes == {MEAN: 3.0, MIN: 2.0}
    assert isinstance(rows[0].runtimes[MEAN], float)


def test_a_verdict_missing_its_flags_reads_as_a_failure(tmp_path):
    rows, _ = labeled(one_unit(tmp_path, [attempt(1, 0)], {"1": [{"sample_id": 0}]}))
    assert (rows[0].compiled, rows[0].correct) == (False, False)


def test_blank_lines_in_attempts_are_skipped(tmp_path):
    run_dir = tmp_path / "myrun"
    write_round(run_dir, "shard_00", 0, *ONE)
    path = run_dir / "shard_00" / "traces" / "round_0" / "attempts.jsonl"
    path.write_text("\n" + path.read_text() + "\n  \n")
    found, _ = units([str(run_dir)], [0])
    rows, counts = labeled(found[0])
    assert len(rows) == 1 and counts == Counter()


def test_the_unit_travels_with_every_row(tmp_path):
    unit = one_unit(tmp_path, [attempt(1, 0), attempt(1, 1)], {"1": [verdict(0), verdict(1)]})
    rows, _ = labeled(unit)
    assert all(r.unit is unit for r in rows)


def test_the_prompt_and_completion_are_copied_verbatim(tmp_path):
    raw = "## Plan\nfirst\n\n```python\nx = 1\n```\n"
    unit = one_unit(tmp_path, [attempt(1, 0, raw=raw, prompt="P!")], {"1": [verdict(0)]})
    rows, _ = labeled(unit)
    assert (rows[0].prompt, rows[0].raw) == ("P!", raw)


# --- the live-corpus rule: half-written JSON is corruption, not a skip (PLAN §13) -----


def test_a_half_written_eval_file_is_fatal_and_names_itself(tmp_path):
    run_dir = tmp_path / "myrun"
    write_round(run_dir, "shard_00", 0, *ONE)
    path = run_dir / "shard_00" / "rounds" / "round_0" / "eval_results.json"
    path.write_text(json.dumps({"1": [verdict(0)]})[:-3])
    found, _ = units([str(run_dir)], [0])
    with pytest.raises(ValueError, match="eval_results.json"):
        labeled(found[0])


def test_a_half_written_attempts_line_is_fatal_and_names_the_line(tmp_path):
    run_dir = tmp_path / "myrun"
    write_round(run_dir, "shard_00", 0, [attempt(1, 0), attempt(1, 1)], {"1": [verdict(0)]})
    path = run_dir / "shard_00" / "traces" / "round_0" / "attempts.jsonl"
    path.write_text(path.read_text()[:-20] + "\n")
    found, _ = units([str(run_dir)], [0])
    with pytest.raises(ValueError, match=r"attempts\.jsonl:2"):
        labeled(found[0])


@pytest.mark.parametrize(
    ("verdicts", "match"),
    [
        ({"1": [{"compiled": True}]}, "sample_id"),
        ({"1": {"sample_id": 0}}, "string indices"),
        ({"one": [verdict(0)]}, "invalid literal"),
        ({"1": [verdict(0), verdict(0)]}, r"two verdicts for \(1, 0\)"),
        ([verdict(0)], "no attribute 'items'"),
        ("oops", "no attribute 'items'"),
    ],
    ids=[
        "no_sample_id",
        "dict_not_list",
        "unparseable_problem_id",
        "duplicate_key",
        "list_not_dict",
        "string_not_dict",
    ],
)
def test_a_malformed_eval_entry_names_the_file_rather_than_raising_a_bare_python_error(
    tmp_path, verdicts, match
):
    # A KeyError out of a pool worker says nothing about which of 127 files produced it.
    # A repeated key is fatal too: there is no way to know which verdict is the sample's.
    unit = one_unit(tmp_path, [attempt(1, 0)], verdicts)
    with pytest.raises(ValueError, match=match) as caught:
        labeled(unit)
    assert "eval_results.json" in str(caught.value)


@pytest.mark.parametrize("name", ["problem_id", "sample_id", "stem", "level", "prompt", "raw"])
def test_an_attempt_record_missing_a_field_names_the_file_and_line(tmp_path, name):
    # The attempts side owes the same attribution as the eval side: a bare KeyError out of
    # a pool worker names neither the file nor the record it came from.
    rec = attempt(1, 0)
    rec.pop(name)
    unit = one_unit(tmp_path, [attempt(1, 0), rec], {"1": [verdict(0)]})
    with pytest.raises(ValueError, match=rf"attempts\.jsonl:2: '{name}'"):
        labeled(unit)


@pytest.mark.parametrize(
    ("rec", "match"),
    [([1, 2], "list indices"), ({"problem_id": "one", "sample_id": 0}, "invalid literal")],
    ids=["line_not_an_object", "unparseable_problem_id"],
)
def test_an_unusable_attempt_record_names_the_file_and_line(tmp_path, rec, match):
    unit = one_unit(tmp_path, [rec], {"1": [verdict(0)]})
    with pytest.raises(ValueError, match=match) as caught:
        labeled(unit)
    assert "attempts.jsonl:1" in str(caught.value)


def test_a_broken_counts_argument_is_not_reported_as_corrupt_data(tmp_path):
    # The ledger bump sits outside the corruption guard: passing a plain dict is the
    # caller's bug, and blaming an innocent line of a healthy file sends the next person
    # -- build.py's author -- looking for a data problem that does not exist.
    unit = one_unit(tmp_path, [attempt(1, 0)], {"1": []})
    with pytest.raises(KeyError):
        list(iter_labeled(unit, {}))


def test_the_fatal_message_says_why_it_is_fatal(tmp_path):
    # A counted skip here would hide the one thing the never-build-on-a-live-eval rule
    # exists to prevent, so the error has to point at the rule rather than at JSON.
    run_dir = tmp_path / "myrun"
    write_round(run_dir, "shard_00", 0, *ONE)
    (run_dir / "shard_00" / "rounds" / "round_0" / "eval_results.json").write_text("{")
    found, _ = units([str(run_dir)], [0])
    with pytest.raises(ValueError, match="never finished"):
        labeled(found[0])


# --- the seam into targets ------------------------------------------------------------


def test_the_runtimes_dict_grades_without_a_further_translation(tmp_path):
    # PRM-4: both sides of the ratio carry the same two keys, so `ours=` takes the row as
    # corpus emits it. A build that had to re-map here could re-introduce the asymmetry.
    entry = verdict(0, runtime=1.0, min_runtime=0.5)
    rows, _ = labeled(one_unit(tmp_path, [attempt(1, 0)], {"1": [entry]}))
    target = target_for(
        compiled=rows[0].compiled,
        correct=rows[0].correct,
        ours=rows[0].runtimes,
        baseline={MEAN: 2.0, MIN: 2.0},
        mode=GRADED,
        stat=MIN,
        lo=0.2,
        hi=4.0,
        quant=0.1,
    )
    assert (target.speedup, target.speedup_min) == (2.0, 4.0)
    assert target.value == pytest.approx(1.0)
