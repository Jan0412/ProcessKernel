"""``prm.build``: the part files, the drop ledger and the manifest (PLAN §10 test_build)."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from reranker.src.prm import build, corpus
from reranker.tests.prm.runfixture import (
    LENGTH,
    STOP,
    attempt,
    baseline_json,
    prm_config,
    verdict,
    write_round,
)

# Four prose cuts at chars 2, 4, 6, 8 -- short enough to reason about by hand.
PROSE = "a\nb\nc\nd\n"
# A fenced block that stops mid-call, so `tokenize` fails and cut_points returns None.
UNTOKENIZABLE = "```python\ndef f(\n```\n"


@pytest.fixture
def chars(monkeypatch):
    """One token per character: keeps the suite off the network and makes max_length exact."""
    monkeypatch.setattr(build, "token_counter", lambda name: len)


def test_the_worker_counts_ids_without_the_special_tokens(monkeypatch):
    # add_special_tokens=False on purpose: the counts are compared against max_length, and
    # the backbone's BOS/EOS are not part of prompt + raw. A stand-in module rather than the
    # real import -- transformers costs 13s to load and would not check anything more.
    seen = {}

    class Fake:
        @staticmethod
        def from_pretrained(name):
            seen["name"] = name
            return lambda text, add_special_tokens: {
                "input_ids": list(text) + ([0] if add_special_tokens else [])
            }

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Fake))
    assert build.token_counter("some/tokenizer")("abc") == 3
    assert seen["name"] == "some/tokenizer"


def a_run(tmp_path, name, attempts, verdicts, *, shard="shard_00"):
    write_round(tmp_path / name, shard, 0, attempts, verdicts)
    return tmp_path / name


def read_json(path):
    with open(path) as f:
        return json.load(f)


def built(tmp_path, attempts, verdicts, **over):
    """Build one run of one unit and return (manifest, parts_dir)."""
    cfg = prm_config(tmp_path, [a_run(tmp_path, "runA", attempts, verdicts)], **over)
    return read_json(build.run_build(cfg)), os.path.join(cfg.prm.out_dir, build.PARTS)


def rows_of(parts_dir, name):
    with open(os.path.join(parts_dir, name)) as f:
        return [json.loads(line) for line in f]


# --- the row -------------------------------------------------------------------------


def test_a_correct_kernel_carries_its_cuts_its_speedups_and_a_graded_target(tmp_path, chars):
    # baseline mean/min 2.0 over a kernel at 1.0 is 2x, which on lo=0.2 hi=4.0 quant=0.1
    # grades p=0.8 -> target (1 + 0.8) / 2.
    manifest, parts = built(
        tmp_path, [attempt(raw=PROSE)], {"0": [verdict(runtime=1.0, min_runtime=1.0)]}
    )
    (row,) = rows_of(parts, "runA__shard_00__round0.jsonl")
    assert row["cuts"] == [2, 4, 6, 8]
    assert row["cut_kinds"] == ["prose"] * 4
    assert row["cut_index"] == [0, 1, 2, 3]
    assert (row["speedup"], row["speedup_min"]) == (2.0, 2.0)
    assert (row["target"], row["label"]) == (0.9, 1)
    assert (row["n_prompt_tokens"], row["n_raw_tokens"]) == (len("PROMPT"), len(PROSE))
    assert (row["run_name"], row["shard"], row["round"]) == ("runA", "shard_00", 0)
    assert (row["level"], row["problem_id"], row["sample_id"]) == (6, 0, 0)
    assert (row["prompt"], row["raw"]) == ("PROMPT", PROSE)
    assert manifest["counts"]["rows"] == 1 and manifest["counts"]["cuts"] == 4


def test_a_lone_surrogate_in_the_completion_is_written_rather_than_fatal(tmp_path, chars):
    # attempts.jsonl stores it escaped, so json.loads hands back a real half-character.
    # Writing that as UTF-8 raises inside a worker and takes the whole pool with it.
    raw = "a\nb\ud800\nc\n"
    _, parts = built(tmp_path, [attempt(raw=raw)], {"0": [verdict(correctness=False)]})
    (row,) = rows_of(parts, "runA__shard_00__round0.jsonl")
    assert row["raw"] == raw


def test_a_wrong_kernel_grades_zero_without_needing_a_baseline(tmp_path, chars):
    _, parts = built(
        tmp_path,
        [attempt(problem_id=99, raw=PROSE)],
        {"99": [verdict(correctness=False)]},
    )
    (row,) = rows_of(parts, "runA__shard_00__round0.jsonl")
    assert (row["target"], row["label"], row["speedup"]) == (0.0, 0, None)


# --- part names and the pool ----------------------------------------------------------


def test_two_runs_that_both_contain_shard_00_write_two_distinct_parts(tmp_path, chars):
    one = a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})
    two = a_run(tmp_path, "runB", [attempt(raw=PROSE)], {"0": [verdict()]})
    cfg = prm_config(tmp_path, [one, two])
    manifest = read_json(build.run_build(cfg))
    assert sorted(manifest["units"]) == [
        "runA__shard_00__round0.jsonl",
        "runB__shard_00__round0.jsonl",
    ]
    assert manifest["counts"]["rows"] == 2


def test_more_than_one_unit_at_more_than_one_worker_goes_through_the_pool(tmp_path, chars):
    one = a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})
    two = a_run(tmp_path, "runB", [attempt(raw=PROSE)], {"0": [verdict()]})
    cfg = prm_config(tmp_path, [one, two], num_workers=2)
    manifest = read_json(build.run_build(cfg))
    assert manifest["units_built"] == 2 and manifest["counts"]["rows"] == 2


# --- the drop ledger ------------------------------------------------------------------


def test_every_drop_reason_fires_and_names_the_stem_it_dropped(tmp_path, chars):
    attempts = [
        attempt(problem_id=1, raw=PROSE, trace=LENGTH),
        attempt(problem_id=2, raw=PROSE, trace=None),
        attempt(problem_id=3, raw=PROSE),  # no verdict below -> no_eval_entry
        attempt(problem_id=4, raw=PROSE),  # correct, and 4 is not in the baseline
        attempt(problem_id=0, raw=PROSE),  # correct, in the baseline, never timed
        attempt(problem_id=6, raw=UNTOKENIZABLE),
        attempt(problem_id=7, raw=PROSE * 40),
        attempt(problem_id=8, raw="no newline, so no cut"),
    ]
    verdicts = {
        "1": [verdict()],
        "2": [verdict()],
        "4": [verdict()],
        "0": [verdict(runtime=-1.0, min_runtime=-1.0)],
        "6": [verdict(correctness=False)],
        "7": [verdict(correctness=False)],
        "8": [verdict(correctness=False)],
    }
    manifest, parts = built(tmp_path, attempts, verdicts, max_length=100)

    assert rows_of(parts, "runA__shard_00__round0.jsonl") == []
    assert manifest["counts"] == {r: 1 for r in build.REASONS} | {"rows": 0}
    assert manifest["attempts"] == len(attempts)
    dropped = manifest["dropped_stems"]
    assert dropped[build.TRUNCATED] == ["level_6_problem_1_sample_0_kernel"]
    assert dropped[build.NO_BASELINE] == ["level_6_problem_4_sample_0_kernel"]
    assert dropped[build.NO_RUNTIME] == ["level_6_problem_0_sample_0_kernel"]
    # no_eval_entry is counted inside corpus.iter_labeled, which never builds a stem.
    assert corpus.NO_EVAL_ENTRY not in dropped


def test_a_truncated_sample_is_dropped_as_truncated_even_when_it_is_unusable_too(tmp_path, chars):
    # Its text would fail to tokenize too. `truncated` must win: it is the only drop that
    # removes a sample because its label is invalid, and a later reason would mask it.
    manifest, _ = built(
        tmp_path,
        [attempt(raw=UNTOKENIZABLE, trace=LENGTH)],
        {"0": [verdict(correctness=False)]},
    )
    assert manifest["counts"][build.TRUNCATED] == 1
    assert build.TOKENIZE_FAILED not in manifest["counts"]


def test_an_unterminated_fence_that_finished_on_its_own_is_kept(tmp_path, chars):
    # The gpt-oss shape -- 99% of that run. A "detect truncation from the fence" refactor
    # must fail here rather than in production.
    manifest, _ = built(
        tmp_path,
        [attempt(raw="prose\n```python\nx = 1\n", trace=STOP)],
        {"0": [verdict(correctness=False)]},
    )
    assert manifest["counts"]["rows"] == 1


# --- min_frac -------------------------------------------------------------------------


def test_min_frac_drops_the_early_cuts_and_leaves_their_depth_alone(tmp_path, chars):
    _, parts = built(
        tmp_path, [attempt(raw=PROSE)], {"0": [verdict(correctness=False)]}, min_frac=0.5
    )
    (row,) = rows_of(parts, "runA__shard_00__round0.jsonl")
    # len(raw) is 8, so the cut at char 2 goes; the survivors keep depths 1, 2, 3.
    assert row["cuts"] == [4, 6, 8]
    assert row["cut_index"] == [1, 2, 3]


def test_min_frac_past_every_cut_drops_the_sample_as_no_cuts(tmp_path, chars):
    # No trailing newline, so the last cut is short of len(raw) and min_frac can pass it.
    manifest, _ = built(
        tmp_path, [attempt(raw="a\nb\nc\nd")], {"0": [verdict(correctness=False)]}, min_frac=0.99
    )
    assert manifest["counts"] == {build.NO_CUTS: 1, "rows": 0}


# --- the sidecar ----------------------------------------------------------------------


def test_the_idx_offsets_seek_to_the_right_rows(tmp_path, chars):
    attempts = [attempt(problem_id=p, raw=PROSE * p) for p in (1, 2, 3)]
    _, parts = built(
        tmp_path, attempts, {str(p): [verdict(correctness=False)] for p in (1, 2, 3)}
    )
    part = os.path.join(parts, "runA__shard_00__round0.jsonl")
    with open(part + ".idx") as f:
        offsets = [int(line) for line in f]
    assert len(offsets) == 3
    with open(part, "rb") as f:
        for offset, expected in zip(offsets, (1, 2, 3)):
            f.seek(offset)
            assert json.loads(f.readline())["problem_id"] == expected


# --- rerun and interruption -----------------------------------------------------------


def test_a_second_invocation_writes_no_new_part_and_keeps_the_ledger(tmp_path, chars):
    # PLAN §7 re-processes only the dropped stems after a tokenizer switch, and that list
    # lives nowhere else -- a rerun to check "did it finish?" must not erase it.
    cfg = prm_config(tmp_path, [a_run(tmp_path, "runA", [attempt(raw=PROSE, trace=LENGTH)],
                                      {"0": [verdict()]})])
    first = read_json(build.run_build(cfg))
    part = os.path.join(cfg.prm.out_dir, build.PARTS, "runA__shard_00__round0.jsonl")
    stamp = os.stat(part).st_mtime_ns
    again = read_json(build.run_build(cfg))
    assert (again["units_built"], again["units_reused"]) == (0, 1)
    assert os.stat(part).st_mtime_ns == stamp
    for key in ("counts", "dropped_stems", "attempts", "units"):
        assert again[key] == first[key], key


def test_the_resume_guard_classifies_every_config_knob():
    # Pinned as a literal, so adding a knob to PRMConfig fails here and has to be classified.
    # Until it is, build.py counts it as row-affecting, which refuses rather than mixes.
    assert set(build.ROW_KNOBS) == {
        "baseline_timing_json",
        "label_mode",
        "speedup_stat",
        "speedup_lo",
        "speedup_hi",
        "speed_quant",
        "prose_lines_per_chunk",
        "code_steps_per_chunk",
        "min_frac",
        "tokenizer",
        "max_length",
    }


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"label_mode": "binary"}, "label_mode"),
        ({"min_frac": 0.5}, "min_frac"),
        ({"code_steps_per_chunk": 5}, "code_steps_per_chunk"),
        ({"max_length": 99}, "max_length"),
        ({"tokenizer": "other/tok"}, "tokenizer"),
    ],
)
def test_resuming_with_a_knob_that_changes_a_row_refuses(tmp_path, chars, over, match):
    # Reused parts carry the old labeling while the manifest records the new config, so
    # one out_dir would hold two labelings with nothing marking which row is which.
    run = a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})
    build.run_build(prm_config(tmp_path, [run]))
    with pytest.raises(ValueError, match=match):
        build.run_build(prm_config(tmp_path, [run], **over))


def test_resuming_after_the_baseline_file_was_remeasured_refuses(tmp_path, chars):
    run = a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})
    build.run_build(prm_config(tmp_path, [run]))
    cfg = prm_config(tmp_path, [run])
    # Same path, different numbers -- every target moves, and only the sha1 sees it.
    baseline_json(tmp_path, problems=((0, 9.0, 9.0),))
    with pytest.raises(ValueError, match="baseline_sha1"):
        build.run_build(cfg)


def test_resuming_with_no_manifest_to_compare_against_refuses(tmp_path, chars):
    # Parts with nothing recording what built them cannot be verified, so reusing them is
    # the one case the guard cannot make safe.
    cfg = prm_config(tmp_path, [a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})])
    build.run_build(cfg)
    os.remove(os.path.join(cfg.prm.out_dir, build.MANIFEST))
    with pytest.raises(ValueError, match="no manifest.json"):
        build.run_build(cfg)


def test_resuming_to_add_a_run_is_still_allowed(tmp_path, chars):
    # What resume is for: the knobs are identical, there is just more corpus.
    one = a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})
    build.run_build(prm_config(tmp_path, [one]))
    two = a_run(tmp_path, "runB", [attempt(raw=PROSE)], {"0": [verdict()]})
    manifest = read_json(build.run_build(prm_config(tmp_path, [one, two])))
    assert (manifest["units_built"], manifest["units_reused"]) == (1, 1)


def test_a_build_killed_before_it_finished_still_leaves_a_manifest(tmp_path, chars):
    # Parts publish one at a time during the pool; the manifest used to be written only
    # after it, so a walltime kill left parts with nothing to check a resume against.
    runs = [
        a_run(tmp_path, "runA", [attempt(raw=PROSE, trace=LENGTH)], {"0": [verdict()]}),
        a_run(tmp_path, "runB", [attempt(raw=PROSE)], {"0": [verdict()]}),
    ]

    def killed(todo, prm, parts_dir, baselines):
        build._init_worker(prm, parts_dir, baselines)
        build._build_one(todo[0])  # one part lands, then the job is gone
        raise KeyboardInterrupt("walltime")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(build, "_map", killed)
    with pytest.raises(KeyboardInterrupt):
        build.run_build(prm_config(tmp_path, runs))
    monkey.undo()

    out_dir = str(tmp_path / "out")
    assert read_json(os.path.join(out_dir, build.MANIFEST))["config"]["max_length"] == 16384
    # that stamp is what lets the resume be checked at all
    with pytest.raises(ValueError, match="max_length"):
        build.run_build(prm_config(tmp_path, runs, max_length=99))
    # and the killed run's part keeps its counters, because they were written beside it
    done = read_json(build.run_build(prm_config(tmp_path, runs)))
    assert (done["units_built"], done["units_reused"], done["parts"]) == (1, 1, 2)
    assert done["counts"][build.TRUNCATED] == 1 and done["counts"]["rows"] == 1


def test_a_part_carries_its_own_counters_so_a_resume_recovers_them(tmp_path, chars):
    run = a_run(tmp_path, "runA", [attempt(raw=PROSE, trace=LENGTH)], {"0": [verdict()]})
    cfg = prm_config(tmp_path, [run])
    build.run_build(cfg)
    name = "runA__shard_00__round0.jsonl"
    meta = read_json(os.path.join(cfg.prm.out_dir, build.PARTS, name + build.META))
    assert meta["dropped"][build.TRUNCATED] == ["level_6_problem_0_sample_0_kernel"]
    # the manifest is the sum of these, so deleting it loses nothing but the config stamp
    assert read_json(os.path.join(cfg.prm.out_dir, build.MANIFEST))["units"][name] == meta


def test_a_sidecar_whose_part_never_landed_is_ignored(tmp_path, chars):
    # build_unit writes .idx and .meta and only then renames the part, so a kill in that
    # window leaves counters for a unit the next run is about to redo.
    cfg = prm_config(tmp_path, [a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})])
    parts = os.path.join(cfg.prm.out_dir, build.PARTS)
    os.makedirs(parts)
    with open(os.path.join(parts, "ghost__shard_00__round0.jsonl" + build.META), "w") as f:
        json.dump({"counts": {"rows": 999}, "dropped": {}}, f)
    manifest = read_json(build.run_build(cfg))
    assert manifest["parts"] == 1 and manifest["counts"]["rows"] == 1


def test_two_identical_builds_produce_the_same_manifest(tmp_path, chars):
    # imap_unordered returns in completion order; if the ledger were merged in that order,
    # the manifest sha would differ between two builds of the same corpus.
    runs = [
        a_run(tmp_path, f"run{c}", [attempt(problem_id=i, raw=PROSE, trace=LENGTH)],
              {str(i): [verdict()]})
        for i, c in enumerate("ABCD")
    ]
    seen = []
    for out in ("one", "two"):
        cfg = prm_config(tmp_path, runs, num_workers=4, out_dir=str(tmp_path / out))
        m = read_json(build.run_build(cfg))
        seen.append({k: v for k, v in m.items() if k not in ("created", "config")})
    assert seen[0] == seen[1]
    assert list(seen[0]["units"]) == sorted(seen[0]["units"])


def test_a_leftover_tmp_is_rebuilt_rather_than_treated_as_done(tmp_path, chars):
    cfg = prm_config(tmp_path, [a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})])
    parts = os.path.join(cfg.prm.out_dir, build.PARTS)
    os.makedirs(parts)
    name = "runA__shard_00__round0.jsonl"
    with open(os.path.join(parts, name + ".tmp"), "w") as f:
        f.write("half a row")
    manifest = read_json(build.run_build(cfg))
    assert manifest["units_built"] == 1
    assert not os.path.exists(os.path.join(parts, name + ".tmp"))
    assert len(rows_of(parts, name)) == 1


# --- provenance -----------------------------------------------------------------------


def test_the_manifest_records_what_produced_the_dataset(tmp_path, chars):
    baseline = baseline_json(tmp_path)
    manifest, _ = built(tmp_path, [attempt(raw=PROSE)], {"0": [verdict()]}, baseline=baseline)
    assert manifest["baseline_timing_json"] == baseline
    assert manifest["baseline_sha1"] == build._sha1(baseline)
    assert manifest["config"]["speedup_stat"] == "min"
    assert manifest["config"]["rounds"] == [0]
    assert manifest["skipped_units"] == []
    assert manifest["created"] and manifest["git_sha"]
    # The sha alone is not provenance -- the prm package is uncommitted as this runs, so
    # HEAD names a commit that does not contain build.py.
    assert manifest["git_dirty"] is True


@pytest.mark.parametrize("blow_up", [OSError("no git here"), subprocess.TimeoutExpired("git", 10)])
def test_a_tree_git_cannot_answer_for_records_no_sha_rather_than_failing(monkeypatch, blow_up):
    def boom(*a, **k):
        raise blow_up

    monkeypatch.setattr(subprocess, "run", boom)
    assert build._git("rev-parse", "HEAD") is None


def test_a_round_with_no_eval_file_is_reported_in_the_manifest(tmp_path, chars):
    run = tmp_path / "runA"
    write_round(run, "shard_00", 0, [attempt(raw=PROSE)], None)
    cfg = prm_config(tmp_path, [run])
    manifest = read_json(build.run_build(cfg))
    assert manifest["skipped_units"] == [["runA", "shard_00", 0, corpus.NO_EVAL]]
    assert manifest["units_built"] == 0


# --- the config gate ------------------------------------------------------------------


def test_a_bad_config_aborts_before_out_dir_exists(tmp_path, chars):
    cfg = prm_config(
        tmp_path,
        [a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})],
        label_mode="binry",
    )
    with pytest.raises(ValueError, match="label_mode"):
        build.run_build(cfg)
    assert not os.path.exists(cfg.prm.out_dir)


def yaml_for(tmp_path, prm, **over):
    path = tmp_path / "c.yaml"
    path.write_text(json.dumps({"prm": dataclasses.asdict(prm) | over}))
    return str(path)


def test_main_builds_the_config_it_was_given(tmp_path, chars):
    cfg = prm_config(tmp_path, [a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})])
    build.main(["--config", yaml_for(tmp_path, cfg.prm)])
    assert read_json(os.path.join(cfg.prm.out_dir, build.MANIFEST))["counts"]["rows"] == 1


def test_main_validates_the_config_it_just_loaded(tmp_path, chars):
    cfg = prm_config(tmp_path, [a_run(tmp_path, "runA", [attempt(raw=PROSE)], {"0": [verdict()]})])
    with pytest.raises(ValueError, match="speedup_stat"):
        build.main(["--config", yaml_for(tmp_path, cfg.prm, speedup_stat="meen")])
    assert not os.path.exists(cfg.prm.out_dir)
