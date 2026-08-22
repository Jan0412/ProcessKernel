"""``data.build_dataset.expand_run_dirs``: the run/shard layout, and the name collision it must not hide."""

from __future__ import annotations

import pytest

from reranker.src.data.build_dataset import expand_run_dirs


def flat(tmp_path, name):
    run = tmp_path / name
    run.mkdir()
    (run / "eval_results.json").write_text("{}")
    return run


def sharded(tmp_path, name, shards, rounds=()):
    run = tmp_path / name
    for s in shards:
        (run / s).mkdir(parents=True)
        (run / s / "eval_results.json").write_text("{}")
        for r in rounds:
            (run / s / "rounds" / f"round_{r}").mkdir(parents=True)
            (run / s / "rounds" / f"round_{r}" / "eval_results.json").write_text("{}")
    return run


def test_a_flat_run_keeps_its_own_basename(tmp_path):
    run = flat(tmp_path, "myrun")
    assert expand_run_dirs([str(run)], [6]) == [(str(run), "myrun", 6)]


def test_a_trailing_slash_does_not_change_the_run_name(tmp_path):
    run = flat(tmp_path, "myrun")
    assert expand_run_dirs([str(run) + "/"], [6])[0][1] == "myrun"


def test_a_sharded_run_expands_to_its_shards_in_order(tmp_path):
    run = sharded(tmp_path, "myrun", ["shard_02", "shard_00", "shard_01"])
    assert expand_run_dirs([str(run)], [6]) == [
        (str(run / "shard_00"), "myrun__shard_00", 6),
        (str(run / "shard_01"), "myrun__shard_01", 6),
        (str(run / "shard_02"), "myrun__shard_02", 6),
    ]


def test_shards_of_two_runs_stay_distinct(tmp_path):
    a = sharded(tmp_path, "runA", ["shard_00"])
    b = sharded(tmp_path, "runB", ["shard_00"])
    names = [n for _, n, _ in expand_run_dirs([str(a), str(b)], [6, 6])]
    assert names == ["runA__shard_00", "runB__shard_00"]


def test_the_level_rides_along_with_each_shard(tmp_path):
    a = sharded(tmp_path, "runA", ["shard_00", "shard_01"])
    b = flat(tmp_path, "runB")
    assert [lvl for _, _, lvl in expand_run_dirs([str(a), str(b)], [6, 5])] == [6, 6, 5]


def test_a_run_dir_that_is_itself_a_shard_still_expands_to_nothing_new(tmp_path):
    run = sharded(tmp_path, "myrun", ["shard_00"])
    shard = run / "shard_00"
    assert expand_run_dirs([str(shard)], [6]) == [(str(shard), "shard_00", 6)]


def test_two_shards_listed_by_hand_collide_on_their_basename(tmp_path):
    # The reason this function exists: <runA>/shard_00 and <runB>/shard_00 are one key
    # downstream, so the second row would overwrite the first instead of joining its list.
    a = sharded(tmp_path, "runA", ["shard_00"])
    b = sharded(tmp_path, "runB", ["shard_00"])
    with pytest.raises(ValueError, match=r"collide on name \['shard_00'\]"):
        expand_run_dirs([str(a / "shard_00"), str(b / "shard_00")], [6, 6])


def test_the_same_run_listed_twice_collides(tmp_path):
    run = sharded(tmp_path, "myrun", ["shard_00"])
    with pytest.raises(ValueError, match="myrun__shard_00"):
        expand_run_dirs([str(run), str(run)], [6, 6])


def test_a_run_with_neither_eval_results_nor_shards_is_skipped_with_a_warning(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    good = flat(tmp_path, "myrun")
    assert expand_run_dirs([str(empty), str(good)], [6, 6]) == [(str(good), "myrun", 6)]


def test_a_run_dir_that_does_not_exist_warns_instead_of_raising(tmp_path, capsys):
    # default.yaml and pairwise_config.yaml still name runs from a cluster that is gone.
    good = flat(tmp_path, "myrun")
    assert expand_run_dirs([str(tmp_path / "gone"), str(good)], [6, 6]) == [(str(good), "myrun", 6)]
    assert "does not exist" in capsys.readouterr().out


def test_a_non_shard_subdirectory_is_not_a_shard(tmp_path):
    run = sharded(tmp_path, "myrun", ["shard_00"])
    (run / "traces").mkdir()
    assert [n for _, n, _ in expand_run_dirs([str(run)], [6])] == ["myrun__shard_00"]


# --- rounds ------------------------------------------------------------------
def test_rounds_read_the_round_dirs_instead_of_the_shard_root(tmp_path):
    run = sharded(tmp_path, "myrun", ["shard_00"], rounds=(0, 1))
    assert expand_run_dirs([str(run)], [6], [0, 1]) == [
        (str(run / "shard_00" / "rounds" / "round_0"), "myrun__shard_00__round0", 6),
        (str(run / "shard_00" / "rounds" / "round_1"), "myrun__shard_00__round1", 6),
    ]


def test_the_shard_root_is_never_read_alongside_its_rounds(tmp_path):
    # The whole point: the root kernel is a byte-identical copy of its own final round, so
    # reading both would enter one kernel twice under two names.
    run = sharded(tmp_path, "myrun", ["shard_00"], rounds=(0, 1, 2))
    dirs = [d for d, _, _ in expand_run_dirs([str(run)], [6], [0, 1, 2])]
    assert str(run / "shard_00") not in dirs
    assert len(dirs) == 3


def test_a_round_a_shard_never_needed_is_skipped_not_fatal(tmp_path):
    run = sharded(tmp_path, "myrun", ["shard_00", "shard_01"], rounds=(0,))
    (run / "shard_00" / "rounds" / "round_1").mkdir()
    (run / "shard_00" / "rounds" / "round_1" / "eval_results.json").write_text("{}")
    names = [n for _, n, _ in expand_run_dirs([str(run)], [6], [0, 1])]
    assert names == ["myrun__shard_00__round0", "myrun__shard_00__round1", "myrun__shard_01__round0"]


def test_the_same_round_of_two_runs_stays_distinct(tmp_path):
    a = sharded(tmp_path, "runA", ["shard_00"], rounds=(0,))
    b = sharded(tmp_path, "runB", ["shard_00"], rounds=(0,))
    names = [n for _, n, _ in expand_run_dirs([str(a), str(b)], [6, 6], [0])]
    assert names == ["runA__shard_00__round0", "runB__shard_00__round0"]


def test_a_repeated_round_collides_rather_than_doubling_every_candidate(tmp_path):
    run = sharded(tmp_path, "myrun", ["shard_00"], rounds=(0,))
    with pytest.raises(ValueError, match="myrun__shard_00__round0"):
        expand_run_dirs([str(run)], [6], [0, 0])


def test_an_unsharded_run_still_expands_to_its_rounds(tmp_path):
    run = tmp_path / "myrun"
    (run / "rounds" / "round_0").mkdir(parents=True)
    (run / "rounds" / "round_0" / "eval_results.json").write_text("{}")
    (run / "eval_results.json").write_text("{}")
    assert expand_run_dirs([str(run)], [6], [0]) == [
        (str(run / "rounds" / "round_0"), "myrun__round0", 6)
    ]
