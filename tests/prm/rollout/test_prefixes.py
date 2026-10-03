"""``prm.prefixes``: v1's parts -> the cut prefixes a v2 campaign will measure (PLAN_v2 §6).

The invariants under test are §4's. N2: a list never spans two runs. N3: val selection is unbiased, whatever the config says. N4: one cut depth per list.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter, defaultdict

import pytest
import yaml

from processkernel.config import PRMRolloutConfig, RerankerConfig
from processkernel.prm.data import build
from processkernel.prm.rollout import prefixes

RUN = "a_run"
TAG = "ar"


def row(*, run=RUN, shard="shard_00", level=6, pid=1, sid=0, n_cuts=20, first=0):
    """One v1 part row, carrying only what prefixes.py reads plus the fields it ignores.

    ``first`` is what prm.min_frac leaves behind: the surviving cuts keep the depth
    cut_points gave them, so cut_index is a *suffix* of 0..n-1, not 0..len-1.
    """
    idx = list(range(first, n_cuts))
    return {
        "run_name": run,
        "shard": shard,
        "level": level,
        "problem_id": pid,
        "sample_id": sid,
        "stem": f"level_{level}_problem_{pid}_sample_{sid}_kernel",
        "system_prompt_sha1": "0" * 40,
        "prompt": "PROMPT",
        "raw": "R" * (10 * n_cuts + 5),
        "cuts": [10 * (i + 1) for i in idx],
        "cut_kinds": ["code" if i % 2 else "prose" for i in idx],
        "cut_index": idx,
        "n_prompt_tokens": 6,
        "n_raw_tokens": 10 * n_cuts,
        "compiled": True,
        "correct": False,
        "speedup": None,
        "speedup_min": None,
        "target": 0.0,
        "label": 0,
    }


def group(n_samples=4, **over):
    """One (run, level, problem) group: ``n_samples`` siblings of the same problem."""
    return [row(sid=s, **over) for s in range(n_samples)]


def cfg(**over) -> PRMRolloutConfig:
    """A section that validates, so each test changes exactly the one thing it is about."""
    base = PRMRolloutConfig(
        run_tags={RUN: TAG},
        baseline_timing_json=__file__,
        min_list_size=2,
        max_list_size=8,
    )
    for k, v in over.items():
        setattr(base, k, v)
    base.validate()
    return base


def emit(rows, conf=None, sp=None, counts=None):
    conf = conf or cfg()
    sp = sp if sp is not None else {(6, 1): "train"}
    return list(prefixes.enumerate_cut_prefixes(rows, conf, sp, counts))


# --- N3: the val split is unbiased, and the type is what enforces it -------------------


def test_a_val_prefix_cannot_be_constructed_with_a_score():
    # In __post_init__ rather than in the caller: N3 is the one invariant a future code
    # path must not be able to violate, and a config knob must be unable to reach it.
    fields = dict(_prefix_fields(), split="val", selection="prm_spread", selection_score=0.8)
    with pytest.raises(ValueError, match="N3"):
        prefixes.Prefix(**fields)
    with pytest.raises(ValueError, match="N3"):
        prefixes.Prefix(**dict(fields, selection="random"))  # score alone is enough
    prefixes.Prefix(**dict(fields, selection="random", selection_score=None))


def test_val_selection_stays_random_when_the_config_asks_for_a_scored_mode():
    # The short-circuit has to happen *before* any scorer is consulted, or a deferred mode
    # would raise NotImplementedError on a split it is forbidden to touch anyway.
    out = emit(group(), conf=cfg(train_selection="prm_spread", prm_v1_checkpoint="/ckpt"),
               sp={(6, 1): "val"})
    assert out
    assert {p.selection for p in out} == {"random"}
    assert {p.selection_score for p in out} == {None}


def test_a_train_problem_under_a_deferred_selection_mode_raises():
    # entropy and prm_spread validate but are not built (§6, decided 2026-08-12).
    conf = cfg(train_selection="prm_spread", prm_v1_checkpoint="/ckpt")
    with pytest.raises(NotImplementedError, match="prm_spread"):
        emit(group(), conf=conf, sp={(6, 1): "train"})


def test_beam_prefixes_are_stage_two_and_raise():
    with pytest.raises(NotImplementedError, match="beam"):
        emit(group(), conf=cfg(source="beam"))


# --- N2 and N4: what a list is --------------------------------------------------------


def test_every_prefix_in_a_list_shares_one_cut_index():
    out = emit(group(n_samples=4))
    by_list = defaultdict(set)
    for p in out:
        by_list[p.list_key].add(p.cut_index)
    assert by_list and all(len(v) == 1 for v in by_list.values())


def test_the_list_key_carries_the_run_the_problem_and_the_depth():
    out = emit(group(pid=37, level=2), sp={(2, 37): "train"})
    assert out
    for p in out:
        assert p.list_key == f"{TAG}:2:37:{p.cut_index}"


# What the paper campaign's code picked for this input: a rebuild from the paper's runs
# must pick the same prefixes.
PAPER_PICKS = [
    (1, 4, 2), (1, 4, 3), (1, 4, 4), (1, 4, 8), (1, 4, 10), (1, 19, 0), (1, 19, 3),
    (1, 19, 6), (1, 19, 7), (1, 19, 11), (1, 26, 1), (1, 26, 3), (1, 26, 4), (1, 26, 8),
    (1, 26, 9), (1, 28, 1), (1, 28, 2), (1, 28, 4), (1, 28, 10), (1, 28, 11), (2, 12, 1),
    (2, 12, 2), (2, 12, 5), (2, 12, 8), (2, 12, 10), (2, 16, 0), (2, 16, 2), (2, 16, 4),
    (2, 16, 7), (2, 16, 10), (2, 27, 0), (2, 27, 1), (2, 27, 3), (2, 27, 6), (2, 27, 8),
    (2, 34, 2), (2, 34, 6), (2, 34, 7), (2, 34, 9), (2, 34, 10), (3, 8, 4), (3, 8, 6),
    (3, 8, 7), (3, 8, 9), (3, 8, 11), (3, 12, 0), (3, 12, 3), (3, 12, 6), (3, 12, 8),
    (3, 12, 11), (3, 20, 1), (3, 20, 2), (3, 20, 7), (3, 20, 8), (3, 20, 10), (3, 31, 1),
    (3, 31, 2), (3, 31, 5), (3, 31, 10), (3, 31, 11),
]


def test_the_draws_are_the_paper_campaigns():
    rows = [r for pid in (1, 2, 3) for r in group(n_samples=12, pid=pid, n_cuts=40)]
    out = emit(rows, conf=cfg(max_list_size=5),
               sp={(6, 1): "train", (6, 2): "val", (6, 3): "train"})
    assert sorted((p.problem_id, p.cut_index, p.sample_id) for p in out) == PAPER_PICKS


def test_two_runs_never_share_a_list_key():
    conf = cfg(run_tags={RUN: TAG, "b_run": "br"})
    rows = group() + group(run="b_run")
    lists = _by_list(emit(rows, conf=conf))
    assert lists
    assert all(len({p.run_name for p in g}) == 1 for g in lists.values())
    assert {k.split(":")[0] for k in lists} == {TAG, "br"}


def test_a_prefix_id_is_unique_and_names_the_run():
    rows = group(n_samples=4) + group(n_samples=4, pid=2)
    out = emit(rows, sp={(6, 1): "train", (6, 2): "train"})
    assert len({p.prefix_id for p in out}) == len(out)
    assert all(p.prefix_id.startswith(f"{TAG}__shard00__") for p in out)


# --- depth: the ruler, the window, and what it is allowed to touch ---------------------


def test_rel_depth_is_the_cut_over_the_completions_own_total_and_never_reaches_one():
    out = emit(group(n_cuts=20))
    assert out
    for p in out:
        assert p.n_cuts_total == 20
        assert p.rel_depth == pytest.approx(p.cut_index / 20)
        assert p.rel_depth < 1.0


def test_a_trimmed_row_still_reports_the_completions_total_not_its_kept_count():
    # prm.min_frac drops leading cuts; the depth ruler is the completion, not the survivors.
    out = emit([row(sid=0, n_cuts=20, first=5), row(sid=1, n_cuts=20, first=5)])
    assert out and {p.n_cuts_total for p in out} == {20}


def test_every_emitted_prefix_lands_inside_the_configured_depth_window():
    rows = [r for pid in range(1, 40) for r in group(pid=pid, n_cuts=10 + pid)]
    sp = {(6, pid): "train" for pid in range(1, 40)}
    out = emit(rows, sp=sp)
    assert out
    assert all(0.1 <= p.rel_depth <= 0.9 for p in out)


def test_the_depth_window_narrows_when_the_config_narrows_it():
    rows = [r for pid in range(1, 40) for r in group(pid=pid, n_cuts=10 + pid)]
    sp = {(6, pid): "train" for pid in range(1, 40)}
    out = emit(rows, conf=cfg(min_rel_depth=0.4, max_rel_depth=0.6), sp=sp)
    assert out
    assert all(0.4 <= p.rel_depth <= 0.6 for p in out)


def test_rel_depth_spreads_over_the_window_rather_than_piling_on_a_few_depths():
    # Uniform over rel_depth, not over k: completions differ in length by 3-4x, so uniform
    # over k would over-sample the shallow region of the long ones. Stratified sampling
    # per group is what makes the aggregate flat instead of `depths_per_group` spikes.
    rows = [r for pid in range(1, 120) for r in group(pid=pid, n_cuts=30 + pid)]
    sp = {(6, pid): "train" for pid in range(1, 120)}
    depths = [p.rel_depth for p in emit(rows, sp=sp)]
    assert len(depths) > 500
    buckets = Counter(min(int((d - 0.1) / 0.8 * 4), 3) for d in depths)
    assert set(buckets) == {0, 1, 2, 3}, "an empty depth bucket means the report cannot slice"
    assert max(buckets.values()) < 2 * min(buckets.values())


def test_a_group_too_shallow_to_cut_inside_the_window_emits_nothing():
    assert emit(group(n_cuts=1)) == []


# --- width: bounded at both ends, and in that order ------------------------------------


def test_a_list_narrower_than_the_floor_is_dropped_whole():
    # One sample cannot form a pair, so its K evals would buy nothing.
    assert emit(group(n_samples=1)) == []
    assert emit(group(n_samples=2)) != []


def test_a_list_wider_than_the_ceiling_is_capped_to_it():
    out = _by_list(emit(group(n_samples=25), conf=cfg(max_list_size=8)))
    assert out and all(len(g) == 8 for g in out.values())


def test_the_cap_is_seeded_so_two_runs_of_the_same_campaign_agree():
    rows = group(n_samples=25)
    first = [p.prefix_id for p in emit(rows)]
    assert first == [p.prefix_id for p in emit(rows)]
    moved = [p.prefix_id for p in emit(rows, conf=cfg(select_seed=43))]
    assert moved != first


def test_the_depth_filter_runs_before_the_cap_never_after():
    # Capping first would let an out-of-window sample take one of the `max_list_size` slots
    # and then be filtered out -- the cap would be deciding which depths survive, and the
    # lists would come out narrower than the ceiling for no stated reason.
    rows = []
    sp = {}
    for pid in range(1, 25):
        # 6 deep siblings and 6 shallow ones: at any k the shallow group's rel_depth is
        # ~6x the deep group's, so a window can hold one and not the other.
        rows += [row(pid=pid, sid=s, n_cuts=600) for s in range(6)]
        rows += [row(pid=pid, sid=6 + s, n_cuts=100) for s in range(6)]
        sp[(6, pid)] = "train"
    lists = _by_list(emit(rows, conf=cfg(max_list_size=4), sp=sp))
    assert lists
    for g in lists.values():
        assert all(0.1 <= p.rel_depth <= 0.9 for p in g)
        assert len(g) == 4


def test_a_sample_without_a_cut_at_the_chosen_depth_is_simply_absent():
    # Lists are ragged by design: a short completion has no cut at a deep k, and dropping
    # the whole list for that would throw away the depths the long ones do reach.
    rows = [row(sid=0, n_cuts=200), row(sid=1, n_cuts=200), row(sid=2, n_cuts=12)]
    lists = _by_list(emit(rows))
    assert lists
    assert any(len(g) == 2 for g in lists.values())
    assert all(len(g) <= 3 for g in lists.values())


# --- splits: read from v1, never recomputed -------------------------------------------


def test_a_problem_missing_from_the_splits_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="6:1"):
        emit(group(), sp={(6, 2): "train"})


def test_a_test_split_problem_contributes_nothing():
    # v1 holds back a test split; a v2 campaign that trained on it would be scoring itself.
    assert emit(group(), sp={(6, 1): "test"}) == []


def test_the_split_lands_on_every_prefix_of_its_problem():
    rows = group(pid=1) + group(pid=2)
    sp = {(6, 1): "train", (6, 2): "val"}
    out = emit(rows, sp=sp)
    assert {p.split for p in out} == {"train", "val"}
    assert all(p.split == sp[(6, p.problem_id)] for p in out)


# --- rows the campaign is not about ----------------------------------------------------


def test_a_row_from_an_untagged_run_is_skipped_not_tagged():
    # run_tags is the run filter as well: one parts dir holds every run v1 built.
    counts = Counter()
    assert emit(group(run="other_run"), counts=counts) == []
    assert counts["row_run_not_selected"] == 4


def test_the_rollout_budget_travels_on_every_row():
    # values.py reads K and min_rollouts off the prefix, not off a config it never sees.
    out = emit(group(), conf=cfg(K=7, min_rollouts=4))
    assert out and {(p.K, p.min_rollouts) for p in out} == {(7, 4)}


def test_the_cut_kind_comes_from_the_chunk_at_that_depth():
    # rollout.py picks its continuation mode off it: prose is a two-pass generation.
    out = emit(group(n_cuts=20))
    assert out
    for p in out:
        assert p.cut_kind == ("code" if p.cut_index % 2 else "prose")
        assert p.cut_char == 10 * (p.cut_index + 1)


# --- the campaign: v1's parts on disk -> prefixes.jsonl --------------------------------


def campaign(tmp_path, parts, sp, **over):
    """Write v1-shaped parts and splits under ``tmp_path``; return a config that reads them."""
    v1 = tmp_path / "prm"
    (v1 / "parts").mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        (v1 / "parts" / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (v1 / "manifest.json").write_text(json.dumps({"git_sha": "v1sha", "counts": {"rows": 1}}))
    (v1 / "splits.json").write_text(json.dumps({f"{lv}:{p}": s for (lv, p), s in sp.items()}))

    over.setdefault("run_tags", {RUN: TAG})
    cfg = RerankerConfig()
    cfg.prm_rollout = PRMRolloutConfig(
        parts_glob=str(v1 / "parts" / "*.jsonl"),
        splits_json=str(v1 / "splits.json"),
        baseline_timing_json=__file__,
        out_dir=str(tmp_path / "out"),
        **over,
    )
    return cfg


def one_part(rows, run=RUN, shard="shard_00"):
    return {f"{run}__{shard}.jsonl": rows}


def written(cfg):
    path = os.path.join(cfg.prm_rollout.out_dir, prefixes.PREFIXES)
    with open(path) as f:
        return [json.loads(line) for line in f]


def manifest_of(cfg):
    with open(os.path.join(cfg.prm_rollout.out_dir, build.MANIFEST)) as f:
        return json.load(f)


def test_the_campaign_writes_one_row_per_prefix(tmp_path):
    cfg = campaign(tmp_path, one_part(group(n_samples=4)), {(6, 1): "train"})
    prefixes.build_prefixes(cfg)
    rows = written(cfg)
    assert rows
    assert len(rows) == manifest_of(cfg)["prefixes"]


def test_the_written_rows_carry_exactly_the_fields_the_contract_names(tmp_path):
    # PLAN_v2 §5. Job B joins on prefix_id and reads cut_char/cut_kind; a field quietly
    # added or dropped here is a contract change no downstream job would notice.
    cfg = campaign(tmp_path, one_part(group()), {(6, 1): "train"})
    prefixes.build_prefixes(cfg)
    assert set(written(cfg)[0]) == {
        "prefix_id", "source", "run_name", "run_tag", "shard", "level",
        "problem_id", "sample_id", "stem", "cut_char", "cut_index", "cut_kind",
        "n_cuts_total", "rel_depth", "list_key", "split", "selection", "selection_score",
        "K", "min_rollouts", "beam_text",
    }


def test_a_part_missing_a_field_job_a_reads_names_itself(tmp_path):
    # The bare KeyError out of the projection names the field but not which of 144 parts
    # carries it -- the one attribution corpus.py makes a rule of, and the reason stats.py
    # guards the same class of thing at its own read (stats.py, scan_parts).
    rows = group(n_samples=2)
    for r in rows:
        del r["cuts"]
    cfg = campaign(tmp_path, one_part(rows), {(6, 1): "train"})
    with pytest.raises(ValueError) as caught:
        prefixes.build_prefixes(cfg)
    assert f"{RUN}__shard_00.jsonl" in str(caught.value)
    assert "cuts" in str(caught.value)


def test_a_glob_that_matches_no_part_raises(tmp_path):
    cfg = campaign(tmp_path, {}, {(6, 1): "train"})
    with pytest.raises(FileNotFoundError, match="no v1 parts"):
        prefixes.build_prefixes(cfg)


def test_a_tag_naming_a_run_no_part_carries_raises(tmp_path):
    # run_tags is the run filter, so a typo'd run name would otherwise select nothing and
    # the campaign would report an empty, entirely valid-looking dataset.
    cfg = campaign(tmp_path, one_part(group()), {(6, 1): "train"},
                   run_tags={RUN: TAG, "typo_run": "tr"})
    with pytest.raises(ValueError, match="typo_run"):
        prefixes.build_prefixes(cfg)


def test_a_group_split_across_two_parts_raises_rather_than_being_cut_twice(tmp_path):
    # The campaign streams one part at a time because a part holds whole groups. If that
    # ever stopped being true, each half would be cut against its own median and emit its
    # own lists under the same list_key -- checked, not assumed.
    parts = {
        f"{RUN}__shard_00.jsonl": group(n_samples=2),
        f"{RUN}__shard_01.jsonl": [row(sid=2), row(sid=3)],
    }
    cfg = campaign(tmp_path, parts, {(6, 1): "train"})
    with pytest.raises(ValueError, match="two parts"):
        prefixes.build_prefixes(cfg)


def test_one_tag_covering_two_levels_raises_because_the_prefix_id_omits_the_level(tmp_path):
    # The tag carries the level by convention (`ds-l2`), which is why prefix_id does not.
    # A run spanning levels would collide two problems' ids into one.
    rows = group(level=1) + [row(level=2, sid=s) for s in range(4)]
    cfg = campaign(tmp_path, one_part(rows), {(1, 1): "train", (2, 1): "train"})
    with pytest.raises(ValueError, match="levels"):
        prefixes.build_prefixes(cfg)


def test_rerunning_the_campaign_writes_the_same_bytes(tmp_path):
    rows = [r for pid in range(1, 12) for r in group(n_samples=12, pid=pid, n_cuts=30 + pid)]
    sp = {(6, pid): "train" for pid in range(1, 12)}
    cfg = campaign(tmp_path, one_part(rows), sp)
    prefixes.build_prefixes(cfg)
    first = open(os.path.join(cfg.prm_rollout.out_dir, prefixes.PREFIXES), "rb").read()
    prefixes.build_prefixes(cfg)
    assert open(os.path.join(cfg.prm_rollout.out_dir, prefixes.PREFIXES), "rb").read() == first
    assert len(first.splitlines()) > 100


@pytest.mark.parametrize("buckets", [2, 4])
def test_the_manifest_records_the_counters_the_widths_and_the_depth_histogram(tmp_path, buckets):
    rows = [r for pid in range(1, 30) for r in group(n_samples=5, pid=pid, n_cuts=40 + pid)]
    sp = {(6, pid): ("val" if pid % 5 == 0 else "train") for pid in range(1, 30)}
    cfg = campaign(tmp_path, one_part(rows), sp, depth_buckets=buckets)
    prefixes.build_prefixes(cfg)
    man = manifest_of(cfg)

    assert man["counts"]["rows_read"] == len(rows)
    assert man["counts"]["groups"] == 29
    assert man["prefixes"] == sum(man["splits"].values()) == len(written(cfg))
    assert man["splits"]["train"] and man["splits"]["val"]
    # As many bands as the report will slice into, and every one of them populated -- an
    # empty band is a per-depth number §6 promises and cannot compute.
    assert len(man["rel_depth_histogram"]) == buckets
    assert all(n > 0 for n in man["rel_depth_histogram"].values())
    assert set(man["list_width_histogram"]) == {"5"}


def test_a_run_the_campaign_does_not_read_is_not_held_to_its_layout(tmp_path):
    # run_tags is the run filter, and one parts dir holds every run v1 built. How a run
    # this campaign never reads lays its problems out is not this campaign's business --
    # otherwise adding a second run to the corpus could fail a campaign about the first.
    parts = {
        f"{RUN}__shard_00.jsonl": group(),
        "other__shard_00.jsonl": [row(run="other", level=1, sid=0)],
        "other__shard_01.jsonl": [row(run="other", level=2, sid=1)],
    }
    cfg = campaign(tmp_path, parts, {(6, 1): "train"})
    prefixes.build_prefixes(cfg)
    assert {r["run_name"] for r in written(cfg)} == {RUN}


def test_the_manifest_says_which_v1_build_it_read_and_which_code_read_it(tmp_path):
    # Without it a prefixes.jsonl cannot be traced to the corpus behind it, and two
    # campaigns over different v1 builds look identical.
    cfg = campaign(tmp_path, one_part(group()), {(6, 1): "train"})
    prefixes.build_prefixes(cfg)
    man = manifest_of(cfg)
    assert man["v1_manifest_sha1"] and man["baseline_sha1"]
    assert man["config"]["select_seed"] == 42 and man["config"]["max_list_size"] == 8
    assert man["parts"] == 1


def test_a_campaign_runs_from_a_config_file_on_the_command_line(tmp_path):
    cfg = campaign(tmp_path, one_part(group()), {(6, 1): "train"})
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({"prm_rollout": dataclasses.asdict(cfg.prm_rollout)}))
    prefixes.main(["--config", str(path)])
    assert written(cfg)


# --- helpers ---------------------------------------------------------------------------


def _by_list(out):
    by = defaultdict(list)
    for p in out:
        by[p.list_key].append(p)
    return by


def _prefix_fields():
    """Every field of Prefix, valid, so a test can move exactly one."""
    return {
        "prefix_id": "ar__shard00__p1__s0__k005",
        "source": "cut",
        "run_name": RUN,
        "run_tag": TAG,
        "shard": "shard_00",
        "level": 6,
        "problem_id": 1,
        "sample_id": 0,
        "stem": "level_6_problem_1_sample_0_kernel",
        "cut_char": 60,
        "cut_index": 5,
        "cut_kind": "code",
        "n_cuts_total": 20,
        "rel_depth": 0.25,
        "list_key": "ar:6:1:5",
        "split": "train",
        "selection": "random",
        "selection_score": None,
        "K": 5,
        "min_rollouts": 3,
        "beam_text": None,
    }
