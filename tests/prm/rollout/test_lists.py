"""``prm.rollout.lists``: measured prefixes -> the ranked lists a trainer reads (PLAN_v2 §6).

Two invariants are enforced here rather than observed: N2 (one run and one problem per list)
and N4 (one cut depth per list). Both raise, because a list that spans either ranks
conditioning or policy rather than prefix quality, and nothing downstream could tell.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest
import yaml

from processkernel.config import PRMRolloutConfig, RerankerConfig
from processkernel.orm.data.labels import speed_p
from processkernel.prm.rollout import calibrate, lists, prefixes, values

TAG = "ar"
LEVEL, PROBLEM, DEPTH = 2, 37, 20
KEY = f"{TAG}:{LEVEL}:{PROBLEM}:{DEPTH}"


def cfg(**over) -> PRMRolloutConfig:
    base = PRMRolloutConfig(
        run_tags={"a_run": TAG}, baseline_timing_json=__file__, K=4, min_rollouts=1
    )
    for k, v in over.items():
        setattr(base, k, v)
    base.validate()
    return base


def pre(prefix_id="p1", **over) -> prefixes.Prefix:
    fields = dict(
        prefix_id=prefix_id,
        source="cut",
        run_name="a_run",
        run_tag=TAG,
        shard="shard_00",
        level=LEVEL,
        problem_id=PROBLEM,
        sample_id=0,
        stem=f"level_{LEVEL}_problem_{PROBLEM}_sample_0_kernel",
        cut_char=10,
        cut_index=DEPTH,
        cut_kind="code",
        n_cuts_total=80,
        rel_depth=0.25,
        list_key=KEY,
        split="train",
        selection="random",
        selection_score=None,
        K=4,
        min_rollouts=1,
    )
    fields.update(over)
    return prefixes.Prefix(**fields)


def val(prefix_id="p1", *, v_graded=0.5, se_graded=0.2, n_rollouts=4, **over) -> values.Value:
    fields = dict(
        prefix_id=prefix_id,
        K_requested=4,
        n_rollouts=n_rollouts,
        n_dropped={reason: 0 for reason in values.REASONS},
        n_dedup_shared=0,
        n_compiled=n_rollouts,
        n_correct=1,
        v_binary=0.25,
        v_graded=v_graded,
        se_binary=0.2,
        se_graded=se_graded,
        scores=[v_graded] * n_rollouts,
    )
    fields.update(over)
    return values.Value(**fields)


def members(*pairs):
    """``[(prefix, value), ...]`` from ``(prefix_id, v_graded)`` pairs, or full objects."""
    out = []
    for pair in pairs:
        if isinstance(pair[0], prefixes.Prefix):
            out.append(pair)
        else:
            pid, v = pair
            out.append((pre(pid), val(pid, v_graded=v)))
    return out


def built(*pairs, split="train", config=None, counts=None):
    return lists.build_one(
        KEY, members(*pairs), split, config or cfg(), counts if counts is not None else Counter()
    )


def test_rel_is_twice_v_graded_so_the_list_lands_on_the_listwise_ladder():
    row = built(("p1", 0.385), ("p2", 0.0))
    assert [item.rel for item in row.items] == [pytest.approx(0.770), 0.0]


def test_a_prefix_with_one_correct_rollout_carries_that_kernels_own_graded_relevance():
    # The scale identity: v2's mean of one lands exactly where orm/lists.py:145 writes a
    # single correct kernel, which is what keeps the two label sets mergeable.
    conf = cfg()
    su = 2.0
    single = (1.0 + speed_p(su, conf.speedup_lo, conf.speedup_hi, conf.speed_quant)) / 2
    row = lists.build_one(
        KEY, [(pre("p1"), val("p1", v_graded=single, n_rollouts=1)), *members(("p2", 0.0))],
        "train", conf, Counter(),
    )
    assert row.items[0].rel == pytest.approx(
        1.0 + speed_p(su, conf.speedup_lo, conf.speedup_hi, conf.speed_quant)
    )


def test_the_error_bar_stays_on_the_v_scale_that_measured_it():
    # `rel` is doubled for the listwise ladder; `se` is not (§5's example carries 0.224 on
    # both the values row and the item). The confidence weight is a ratio of V̂ differences
    # to V̂ error bars, so mixing the two scales would inflate every weight by 2x.
    row = built((pre("p1"), val("p1", v_graded=0.62, se_graded=0.18)),
                (pre("p2"), val("p2", v_graded=0.30, se_graded=0.20)))
    assert [item.se for item in row.items] == [0.18, 0.20]
    a, b = row.items
    from_items = abs(a.rel / 2 - b.rel / 2) / ((a.se**2 + b.se**2) ** 0.5)
    from_values = abs(0.62 - 0.30) / ((0.18**2 + 0.20**2) ** 0.5)
    assert from_items == pytest.approx(from_values)


def test_a_list_whose_prefixes_all_measured_the_same_is_dropped_and_counted():
    # No valid ranking pair -- the same two lines orm/lists.py:163 already spends.
    counts: Counter = Counter()
    assert built(("p1", 0.5), ("p2", 0.5), counts=counts) is None
    assert counts[lists.ALL_EQUAL] == 1


def test_a_list_below_min_list_size_is_dropped_and_counted():
    counts: Counter = Counter()
    assert built(("p1", 0.5), counts=counts) is None
    assert counts[lists.TOO_SMALL] == 1


def test_a_one_item_list_is_counted_once_and_by_its_size():
    # A one-item list is too small AND trivially all-equal; counting it twice would make the
    # ledger add up to more lists than the campaign built.
    counts: Counter = Counter()
    built(("p1", 0.5), counts=counts)
    assert counts[lists.ALL_EQUAL] == 0


def test_every_item_shares_the_run_problem_and_depth_of_its_key():
    row = built(("p1", 0.8), ("p2", 0.0))
    assert (row.run_tag, row.level, row.problem_id) == (TAG, LEVEL, PROBLEM)
    assert row.cut_index == DEPTH
    assert row.source == "cut"


def test_two_problems_in_one_list_raise():
    # N2: two problems' prefixes sit behind different prompts, so ranking across them ranks
    # conditioning, not the prefix.
    with pytest.raises(ValueError, match="N2"):
        built((pre("p1"), val("p1", v_graded=0.8)), (pre("p2", problem_id=38), val("p2")))


def test_two_runs_in_one_list_raise():
    with pytest.raises(ValueError, match="N2"):
        built((pre("p1"), val("p1", v_graded=0.8)),
              (pre("p2", run_tag="other", run_name="b_run"), val("p2")))


def test_two_cut_depths_in_one_list_raise():
    # N4: equal chunk counts are what make the items comparable states of one problem.
    with pytest.raises(ValueError, match="N4"):
        built((pre("p1"), val("p1", v_graded=0.8)), (pre("p2", cut_index=21), val("p2")))


def test_rel_depth_mean_is_carried_so_the_report_need_not_rejoin_prefixes():
    row = built((pre("p1", rel_depth=0.2), val("p1", v_graded=0.8)),
                (pre("p2", rel_depth=0.4), val("p2")))
    assert row.rel_depth_mean == pytest.approx(0.3)


def test_items_are_ordered_by_prefix_id_so_two_builds_agree():
    row = built(("p2", 0.8), ("p1", 0.0))
    assert [item.prefix_id for item in row.items] == ["p1", "p2"]


def test_an_item_carries_the_n_rollouts_its_value_was_measured_from():
    row = built((pre("p1"), val("p1", v_graded=0.8, n_rollouts=3)), ("p2", 0.0))
    assert row.items[0].n_rollouts == 3


# --- grouping: a prefix with no measured value simply is not in its list -----------------


def test_a_prefix_the_campaign_could_not_measure_leaves_a_ragged_list():
    # `lambdarank_loss` already handles variable group sizes via `group_sizes`, so a member
    # that was dropped upstream shortens the list rather than invalidating it.
    grouped = lists.group([pre("p1"), pre("p2"), pre("p3")], [val("p1"), val("p3")])
    assert [p.prefix_id for p, _ in grouped[KEY]] == ["p1", "p3"]


def test_grouping_orders_each_lists_members_however_the_values_arrive():
    # values.jsonl is written prefix by prefix; nothing downstream should depend on which
    # order that was, so the boundary sorts rather than every consumer of it.
    grouped = lists.group([pre("p1"), pre("p2")], [val("p2"), val("p1")])
    assert [p.prefix_id for p, _ in grouped[KEY]] == ["p1", "p2"]


def test_grouping_keeps_each_list_key_apart():
    other = f"{TAG}:{LEVEL}:{PROBLEM}:21"
    grouped = lists.group(
        [pre("p1"), pre("p2", cut_index=21, list_key=other)], [val("p1"), val("p2")]
    )
    assert set(grouped) == {KEY, other}


def test_a_value_naming_no_prefix_is_an_error():
    # values.jsonl and prefixes.jsonl are written by the same campaign; a value the prefixes
    # do not hold means the two files come from different builds.
    with pytest.raises(KeyError, match="p9"):
        lists.group([pre("p1")], [val("p1"), val("p9")])


# --- the pass: splits.json decides which file a list lands in ----------------------------


def campaign(tmp_path, *, pairs=(("p1", 0.8), ("p2", 0.0)), splits=None, **over):
    out_dir = tmp_path / "campaign"
    out_dir.mkdir()
    pairs = members(*pairs)
    (out_dir / prefixes.PREFIXES).write_text(
        "".join(json.dumps(dataclasses.asdict(p)) + "\n" for p, _ in pairs)
    )
    (out_dir / values.VALUES).write_text(
        "".join(json.dumps(dataclasses.asdict(v)) + "\n" for _, v in pairs)
    )
    splits_path = tmp_path / "splits.json"
    default = {f"{LEVEL}:{PROBLEM}": "train"}
    splits_path.write_text(json.dumps(default if splits is None else splits))
    return RerankerConfig(
        prm_rollout=cfg(out_dir=str(out_dir), splits_json=str(splits_path), **over)
    ), out_dir


def read_lists(out_dir, split) -> list[dict]:
    text = (out_dir / lists.LISTS.format(split=split)).read_text()
    return [json.loads(line) for line in text.splitlines()]


def test_a_list_lands_in_the_file_its_problems_split_names(tmp_path):
    config, out_dir = campaign(tmp_path)
    lists.build_lists(config)
    (row,) = read_lists(out_dir, "train")
    assert row["list_key"] == KEY
    assert [item["prefix_id"] for item in row["items"]] == ["p1", "p2"]
    assert read_lists(out_dir, "val") == []


def test_the_split_comes_from_splits_json_and_not_from_the_prefix_row(tmp_path):
    # The plan's wording (§6): job D reads v1's splits.json. The prefix row carries a `split`
    # too -- job A's copy of the same lookup -- and this pins which of the two decides.
    config, out_dir = campaign(tmp_path, splits={f"{LEVEL}:{PROBLEM}": "val"})
    lists.build_lists(config)
    assert read_lists(out_dir, "train") == []
    assert read_lists(out_dir, "val")[0]["split"] == "val"


def test_a_problem_with_no_split_at_all_raises(tmp_path):
    config, _ = campaign(tmp_path, splits={"2:99": "train"})
    with pytest.raises(KeyError, match=f"{LEVEL}:{PROBLEM}"):
        lists.build_lists(config)


def test_a_problem_v1_held_back_for_test_is_counted_out_of_both_files(tmp_path):
    config, out_dir = campaign(tmp_path, splits={f"{LEVEL}:{PROBLEM}": "test"})
    manifest = lists.build_lists(config)
    assert manifest["dropped"][lists.OTHER_SPLIT] == 1
    assert read_lists(out_dir, "train") == [] and read_lists(out_dir, "val") == []


def test_the_manifest_counts_lists_items_and_the_pairs_among_them(tmp_path):
    config, _ = campaign(tmp_path)
    manifest = lists.build_lists(config)
    assert manifest["lists"]["train"] == 1
    assert manifest["items"]["train"] == 2
    # A corpus that is mostly pairs is a pairwise dataset wearing a listwise schema (§2).
    assert manifest["two_item_lists"]["train"] == 1


def test_the_cli_writes_both_files_even_when_one_is_empty(tmp_path):
    config, out_dir = campaign(tmp_path)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({"prm_rollout": dataclasses.asdict(config.prm_rollout)}))
    lists.main(["--config", str(path)])
    assert (out_dir / lists.LISTS.format(split="val")).exists()
    assert (out_dir / lists.LISTS_MANIFEST).exists()


def test_a_list_the_campaign_cannot_rank_is_counted_rather_than_written(tmp_path):
    config, out_dir = campaign(tmp_path, pairs=(("p1", 0.5), ("p2", 0.5)))
    manifest = lists.build_lists(config)
    assert manifest["dropped"][lists.ALL_EQUAL] == 1
    assert manifest["lists"] == {"train": 0, "val": 0}
    assert read_lists(out_dir, "train") == []


# --- N5: a list never mixes label_source, and imputed items pick up se_imputed -----------
#
# The brief's own two-line sketch (`build_one([_item(src="measured"), ...])`) calls
# `build_one` with only a list of items, but the real function still needs `key`/`split`/
# `cfg`/`counts` (N2/N4/min_list_size all live here too), and `min_list_size` (>= 2, checked
# in `cfg().validate()`) means a genuine one-item call would be dropped as TOO_SMALL before
# N5 is ever reached. Adapted to the real 5-arg signature and two items -- see task-6-report.


def test_a_list_never_mixes_label_sources():
    with pytest.raises(ValueError, match="label_source"):
        built(
            (pre("p1"), val("p1", v_graded=0.8, label_source="measured")),
            (pre("p2"), val("p2", v_graded=0.0, label_source="imputed")),
        )


def test_rel_is_two_v_graded_on_the_imputed_path_too():
    row = built(
        (pre("p1"), val("p1", v_graded=0.385, label_source="imputed", se_imputed=0.11)),
        (pre("p2"), val("p2", v_graded=0.0, label_source="imputed", se_imputed=0.11)),
    )
    assert row.label_source == "imputed"
    assert row.items[0].rel == pytest.approx(0.77)


def test_an_imputed_items_se_is_se_imputed_not_se_graded():
    # One field, two ways of computing it (lists.py's docstring): se_graded would silently
    # carry the DEFAULT from `val()` (0.2) rather than the value actually passed.
    row = built(
        (pre("p1"), val("p1", v_graded=0.8, label_source="imputed", se_imputed=0.05)),
        (pre("p2"), val("p2", v_graded=0.0, label_source="imputed", se_imputed=0.09)),
    )
    assert [item.se for item in row.items] == [0.05, 0.09]


# --- drop_dead_problems: a problem no anchor of which ever passed -------------------------
#
# Its imputed V̂ are all the ORM's invention and so is their order, and the ALL_EQUAL drop
# above cannot see it: imputed V̂ are floats and never tie. The dead set is decided by
# calibrate.py, which is the only job that loads the anchors, and read back here.


def dead_file(out_dir, *pkeys):
    (out_dir / calibrate.OFFSETS).write_text(
        json.dumps({"meta": {}, "offsets": {}, "dead": list(pkeys)}))


def test_a_dead_problems_lists_are_dropped_and_counted(tmp_path):
    config, out_dir = campaign(tmp_path, drop_dead_problems=True)
    dead_file(out_dir, f"{LEVEL}:{PROBLEM}")
    manifest = lists.build_lists(config)
    assert manifest["dropped"][lists.DEAD_PROBLEM] == 1
    assert manifest["lists"] == {"train": 0, "val": 0}
    assert read_lists(out_dir, "train") == []


def test_a_live_problem_is_untouched_by_the_filter(tmp_path):
    config, out_dir = campaign(tmp_path, drop_dead_problems=True)
    dead_file(out_dir, "6:999")
    manifest = lists.build_lists(config)
    assert manifest["dropped"][lists.DEAD_PROBLEM] == 0
    assert len(read_lists(out_dir, "train")) == 1


def test_the_filter_is_off_unless_the_config_asks_for_it(tmp_path):
    # Level 1 is measured and keeps every list; only an imputed campaign turns this on.
    config, out_dir = campaign(tmp_path)
    dead_file(out_dir, f"{LEVEL}:{PROBLEM}")
    manifest = lists.build_lists(config)
    assert manifest["dropped"][lists.DEAD_PROBLEM] == 0
    assert len(read_lists(out_dir, "train")) == 1


def test_the_filter_refuses_to_run_before_job_d_has(tmp_path):
    # Silently dropping nothing would look exactly like a campaign with no dead problems.
    config, _ = campaign(tmp_path, drop_dead_problems=True)
    with pytest.raises(FileNotFoundError, match="calibrate"):
        lists.build_lists(config)


def test_a_dead_list_is_counted_once_and_not_also_as_all_equal(tmp_path):
    # The ledger has to add up: every list leaves by exactly one door.
    config, out_dir = campaign(tmp_path, pairs=(("p1", 0.5), ("p2", 0.5)),
                               drop_dead_problems=True)
    dead_file(out_dir, f"{LEVEL}:{PROBLEM}")
    manifest = lists.build_lists(config)
    assert manifest["dropped"][lists.DEAD_PROBLEM] == 1
    assert manifest["dropped"][lists.ALL_EQUAL] == 0
