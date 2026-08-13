"""``prm.rollout.values``: the eval results a campaign bought -> V̂ per prefix (PLAN_v2 §6, job D).

The invariant under test throughout is N1: every number here comes from the prefix's own
rollouts. v1's verdict for the completion the prefix was cut out of is one draw from a
different distribution, and nothing in this module may reach for it.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter

import pytest
import yaml

from reranker.src.config import PRMRolloutConfig, RerankerConfig
from reranker.src.prm.rollout import prefixes, rollout, stage, values

TAG = "ar"
LEVEL, PROBLEM = 2, 37
# speedup 1.0 against lo=0.2 hi=4.0 quant=0.1 -> speed_p 0.5 -> graded target 0.75.
BASELINES = {LEVEL: {PROBLEM: {"mean": 1.0, "min": 1.0}}}
AT_BASELINE = 0.75


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
        round=0,
        level=LEVEL,
        problem_id=PROBLEM,
        sample_id=0,
        stem=f"level_{LEVEL}_problem_{PROBLEM}_sample_0_kernel",
        cut_char=10,
        cut_index=20,
        cut_kind="code",
        n_cuts_total=80,
        rel_depth=0.25,
        list_key=f"{TAG}:{LEVEL}:{PROBLEM}:0:20",
        split="train",
        selection="random",
        selection_score=None,
        K=4,
        min_rollouts=1,
    )
    fields.update(over)
    return prefixes.Prefix(**fields)


def roll(rid, **over) -> rollout.Rollout:
    fields = dict(
        rollout_id=rid,
        prefix_id=rid.rsplit("__j", 1)[0],
        j=0,
        continuation="...",
        code="import torch\n",
        code_sha1="a" * 40,
        n_prefix_tokens=10,
        n_gen_tokens=20,
        finish_reason={"plan": None, "code": "stop"},
        truncation="ok",
    )
    fields.update(over)
    return rollout.Rollout(**fields)


def verdict(correct=True, compiled=True, runtime=1.0, fastest=1.0) -> dict:
    return {
        "compiled": compiled,
        "correctness": correct,
        "runtime": runtime,
        "runtime_stats": {"min": fastest},
    }


def graded(*outcomes, shared=()) -> tuple:
    """``(rollouts, joined)`` for one prefix: True is a correct kernel at baseline speed."""
    rows = [roll(f"p1__j{j:02d}") for j in range(len(outcomes))]
    joined = {
        r.rollout_id: values.Eval(verdict(correct=c), shared=r.rollout_id in shared)
        for r, c in zip(rows, outcomes)
    }
    return rows, joined


def value(*outcomes, prefix=None, config=None, **over):
    rows, joined = graded(*outcomes, **over)
    counts: Counter = Counter()
    return values.value_for(
        prefix or pre(), rows, joined, BASELINES, config or cfg(), counts
    ), counts


def specs(*rows, prefix=None, config=None, baselines=None):
    """``[{truncation, verdict, shared}, ...]`` -> ``(value_or_None, campaign counts)``."""
    rollouts, joined = [], {}
    for j, spec in enumerate(rows):
        r = roll(f"p1__j{j:02d}", truncation=spec.get("truncation", "ok"))
        rollouts.append(r)
        joined[r.rollout_id] = values.Eval(
            spec["verdict"] if "verdict" in spec else verdict(), spec.get("shared", False)
        )
    counts: Counter = Counter()
    row = values.value_for(
        prefix or pre(),
        rollouts,
        joined,
        BASELINES if baselines is None else baselines,
        config or cfg(),
        counts,
    )
    return row, counts


def test_v_graded_is_the_mean_of_the_rollouts_own_graded_targets():
    row, _ = value(True, True, False, False)
    assert row.v_graded == pytest.approx((AT_BASELINE + AT_BASELINE + 0.0 + 0.0) / 4)


def test_v_binary_is_the_share_of_correct_rollouts():
    row, _ = value(True, True, False, False)
    assert row.v_binary == pytest.approx(0.5)
    assert row.n_correct == 2
    assert row.n_rollouts == 4


def test_se_binary_is_the_binomial_error_of_v_binary():
    row, _ = value(True, True, False, False)
    assert row.se_binary == pytest.approx(math.sqrt(0.5 * 0.5 / 4))


def test_se_graded_is_the_sample_std_of_the_scores_over_sqrt_n():
    row, _ = value(True, True, False, False)
    # stdev([.75, .75, 0, 0]) = 0.4330127, over sqrt(4).
    assert row.se_graded == pytest.approx(0.4330127018922193 / 2)


def test_scores_are_stored_per_rollout_so_v_graded_re_aggregates():
    row, _ = value(True, True, False, False)
    assert row.scores == [AT_BASELINE, AT_BASELINE, 0.0, 0.0]
    assert sum(row.scores) / len(row.scores) == pytest.approx(row.v_graded)


def test_scores_are_ordered_by_rollout_id_whatever_order_they_arrive_in():
    # Job B returns its rollouts bucketed by token budget, so arrival order is not stable
    # across reruns -- and `scores` is a stored column that two runs have to agree on.
    rows = [roll("p1__j01"), roll("p1__j00")]
    joined = {
        "p1__j01": values.Eval(verdict(correct=False)),
        "p1__j00": values.Eval(verdict(correct=True)),
    }
    row = values.value_for(pre(), rows, joined, BASELINES, cfg(), Counter())
    assert row.scores == [AT_BASELINE, 0.0]


def test_a_single_rollout_has_no_measurable_spread():
    row, _ = value(True)
    assert row.n_rollouts == 1
    assert row.se_graded == 0.0
    assert row.se_binary == 0.0


# --- drops: a rollout that cannot be measured leaves K, it does not score zero -----------


def test_a_truncated_rollout_reduces_n_rollouts_rather_than_scoring_zero():
    # v1 §2's reason: a forcibly-stopped generation is not a draw from the policy, and its
    # kernel is a fragment, so it would score 0 however good the prefix was.
    row, counts = specs({}, {}, {"truncation": "truncated"})
    assert row.n_rollouts == 2
    assert row.v_graded == pytest.approx(AT_BASELINE)
    assert row.n_dropped[values.TRUNCATED] == 1
    assert counts[values.TRUNCATED] == 1


def test_a_rollout_with_no_finish_reason_is_dropped_apart_from_a_truncated_one():
    row, _ = specs({}, {}, {"truncation": "unknown"})
    assert row.n_rollouts == 2
    assert row.n_dropped[values.NO_FINISH_REASON] == 1
    assert row.n_dropped[values.TRUNCATED] == 0


def test_a_rollout_the_eval_never_returned_is_counted_not_scored():
    row, _ = specs({}, {}, {"verdict": None})
    assert row.n_rollouts == 2
    assert row.n_dropped[values.NO_EVAL_ENTRY] == 1


def test_truncation_is_decided_before_the_eval_is_looked_up():
    # Both apply; the ledger is only interpretable if the order is fixed (§6).
    row, _ = specs({}, {}, {"truncation": "truncated", "verdict": None})
    assert row.n_dropped[values.TRUNCATED] == 1
    assert row.n_dropped[values.NO_EVAL_ENTRY] == 0


def test_a_correct_kernel_with_no_baseline_is_dropped_as_no_baseline():
    row, counts = specs({}, {}, {}, baselines={})
    # Nothing survives, so the prefix goes too -- but by the reason that emptied it.
    assert row is None
    assert counts[values.NO_BASELINE] == 3
    assert counts[values.NO_RUNTIME] == 0


def test_a_correct_kernel_the_harness_never_timed_is_told_apart_from_a_missing_baseline():
    # PRM-4: a kernel with no runtime is not an absent baseline, and one ledger entry for
    # both would hide an eval that ran and reported nothing.
    row, _ = specs({}, {}, {"verdict": verdict(fastest=None)})
    assert row.n_rollouts == 2
    assert row.n_dropped[values.NO_RUNTIME] == 1
    assert row.n_dropped[values.NO_BASELINE] == 0


def test_a_failed_kernel_needs_no_baseline_at_all():
    row, _ = specs({"verdict": verdict(correct=False)}, {"verdict": verdict(correct=False)},
                   baselines={})
    assert row.n_rollouts == 2
    assert row.v_graded == 0.0


def test_a_prefix_below_min_rollouts_is_dropped_whole_and_counted_once():
    row, counts = specs({}, {"truncation": "truncated"}, {"truncation": "truncated"},
                        config=cfg(min_rollouts=2))
    assert row is None
    assert counts[values.TOO_FEW_ROLLOUTS] == 1
    # The rollout-level drops that got it there are still on the campaign ledger.
    assert counts[values.TRUNCATED] == 2


def test_every_drop_reason_is_on_the_row_so_a_missing_key_is_never_a_zero():
    row, _ = specs({}, {})
    assert row.n_dropped == {reason: 0 for reason in values.REASONS}


def test_a_rollout_the_map_does_not_place_is_an_error_not_a_silent_drop():
    # A rollout generated after the staging pass is in no eval run dir at all; scoring the
    # prefix without it would report a K the campaign never bought.
    rows = [roll("p1__j00")]
    with pytest.raises(KeyError, match="p1__j00"):
        values.value_for(pre(), rows, {}, BASELINES, cfg(), Counter())


def test_a_rollout_that_shared_another_s_eval_is_counted():
    row, _ = specs({}, {"shared": True})
    assert row.n_dedup_shared == 1
    assert row.n_rollouts == 2


# --- the join: every eval shard's results, back through the map stage.py derived ---------


def placed(problem, sample, shared=False, level=LEVEL) -> dict:
    return {"level": level, "problem_id": problem, "sample_id": sample, "shared": shared}


def shard_dir(tmp_path, index, results: dict | None, name="prm_rollout_v1"):
    run_dir = tmp_path / f"{name}_s{index:02d}"
    run_dir.mkdir()
    if results is not None:
        (run_dir / "eval_results.json").write_text(json.dumps(results))
    return run_dir


def test_the_verdicts_of_every_eval_shard_are_read_as_one_table(tmp_path):
    shard_dir(tmp_path, 0, {"37": [{"sample_id": 0, "correctness": True}]})
    shard_dir(tmp_path, 1, {"41": [{"sample_id": 0, "correctness": False}]})
    table = values.read_verdicts(str(tmp_path), "prm_rollout_v1")
    assert set(table) == {(37, 0), (41, 0)}


def test_a_shard_that_has_not_been_evaluated_yet_is_an_error_not_an_empty_table(tmp_path):
    # Silently returning nothing would grade every one of its rollouts `no_eval_entry` and
    # drop their prefixes -- a campaign quietly shrunk to whichever shards happened to finish.
    shard_dir(tmp_path, 0, {"37": [{"sample_id": 0, "correctness": True}]})
    shard_dir(tmp_path, 1, None)
    with pytest.raises(FileNotFoundError, match="_s01 has not been evaluated yet"):
        values.read_verdicts(str(tmp_path), "prm_rollout_v1")


def test_a_campaign_with_no_eval_run_dirs_at_all_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prm_rollout_v1"):
        values.read_verdicts(str(tmp_path), "prm_rollout_v1")


def test_one_sample_graded_by_two_shards_is_an_error(tmp_path):
    # The shard plan partitions the problems, so this means two run dirs were pointed at
    # overlapping subsets and there is no way to say which verdict is the sample's.
    shard_dir(tmp_path, 0, {"37": [{"sample_id": 0, "correctness": True}]})
    shard_dir(tmp_path, 1, {"37": [{"sample_id": 0, "correctness": False}]})
    with pytest.raises(ValueError, match="37"):
        values.read_verdicts(str(tmp_path), "prm_rollout_v1")


def test_a_campaign_whose_name_merely_extends_this_one_is_not_read(tmp_path):
    # `{run_name}_s*` is a prefix match: the repo already names campaigns X and X_smoke
    # (data/prm_rollout, data/prm_rollout_smoke), and both stage dense sample ids from 0 over
    # the same level, so merging them either raises or grades a prefix against another run.
    shard_dir(tmp_path, 0, {"37": [{"sample_id": 0, "correctness": True}]})
    shard_dir(tmp_path, 0, {"99": [{"sample_id": 0, "correctness": True}]},
              name="prm_rollout_v1_smoke")
    assert set(values.read_verdicts(str(tmp_path), "prm_rollout_v1")) == {(37, 0)}


def test_a_file_beside_the_shard_dirs_is_not_mistaken_for_one(tmp_path):
    shard_dir(tmp_path, 0, {"37": [{"sample_id": 0, "correctness": True}]})
    (tmp_path / "prm_rollout_v1_s02.tar").write_text("an archived shard")
    assert set(values.read_verdicts(str(tmp_path), "prm_rollout_v1")) == {(37, 0)}


def test_the_join_gives_each_rollout_the_verdict_of_the_sample_it_was_staged_as():
    rows = [roll("p1__j00"), roll("p1__j01")]
    rmap = {"p1__j00": placed(PROBLEM, 0), "p1__j01": placed(PROBLEM, 1)}
    table = {(PROBLEM, 0): verdict(correct=True), (PROBLEM, 1): verdict(correct=False)}
    joined = values.join(rows, rmap, table)
    assert joined["p1__j00"].verdict["correctness"] is True
    assert joined["p1__j01"].verdict["correctness"] is False


def test_a_deduped_rollout_resolves_through_the_eval_that_was_actually_run():
    rows = [roll("p1__j00"), roll("p1__j01")]
    rmap = {"p1__j00": placed(PROBLEM, 0), "p1__j01": placed(PROBLEM, 0, shared=True)}
    joined = values.join(rows, rmap, {(PROBLEM, 0): verdict(correct=True)})
    assert joined["p1__j01"].verdict["correctness"] is True
    assert joined["p1__j01"].shared is True
    assert joined["p1__j00"].shared is False


def test_a_placement_the_harness_never_graded_joins_to_nothing():
    rows = [roll("p1__j00")]
    joined = values.join(rows, {"p1__j00": placed(PROBLEM, 0)}, {})
    assert joined["p1__j00"].verdict is None


def test_a_rollout_missing_from_the_map_is_an_error():
    # Naming the rollout is not enough -- a bare lookup does that. The error has to say the
    # campaign was re-generated after staging, which is the only way to get here.
    with pytest.raises(KeyError, match="generated after the staging pass"):
        values.join([roll("p1__j00")], {}, {})


# --- the pass: one campaign's parts, map and eval shards -> values.jsonl -----------------


def campaign(tmp_path, *, outcomes=(True, False), prefix_rows=None, rollout_ids=None, **over):
    """A whole small campaign on disk: prefixes, one rollout part, the map, one eval shard."""
    out_dir = tmp_path / "campaign"
    (out_dir / stage.ROLLOUTS).mkdir(parents=True)
    runs = tmp_path / "runs"
    runs.mkdir()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({f"level{LEVEL}": {f"{PROBLEM}_a_problem.py": {"mean": 1.0, "min": 1.0}}})
    )

    rows = [roll(rid) for rid in (rollout_ids or [f"p1__j{j:02d}" for j in range(len(outcomes))])]
    stage.write_rollouts(rows, str(out_dir / stage.ROLLOUTS / "unit.jsonl.gz"))
    (out_dir / prefixes.PREFIXES).write_text(
        "".join(json.dumps(dataclasses.asdict(p)) + "\n" for p in (prefix_rows or [pre()]))
    )
    (out_dir / stage.ROLLOUT_MAP).write_text(
        json.dumps({r.rollout_id: placed(PROBLEM, j) for j, r in enumerate(rows)})
    )
    shard_dir(
        runs,
        0,
        {
            str(PROBLEM): [
                dict(verdict(correct=c), sample_id=j) for j, c in enumerate(outcomes)
            ]
        },
    )
    return RerankerConfig(
        prm_rollout=cfg(
            out_dir=str(out_dir),
            eval_runs_dir=str(runs),
            eval_run_name="prm_rollout_v1",
            baseline_timing_json=str(baseline),
            **over,
        )
    ), out_dir


def written(out_dir) -> list[dict]:
    return [json.loads(line) for line in (out_dir / values.VALUES).read_text().splitlines()]


def test_the_pass_measures_every_prefix_its_campaign_generated_for(tmp_path):
    config, out_dir = campaign(tmp_path)
    values.build_values(config)
    (row,) = written(out_dir)
    assert row["prefix_id"] == "p1"
    assert row["n_rollouts"] == 2
    assert row["v_graded"] == pytest.approx(AT_BASELINE / 2)
    assert row["scores"] == [AT_BASELINE, 0.0]


def test_the_pass_never_opens_v1s_parts(tmp_path):
    # N1, as a property of the pass rather than of one aggregate: the v1 label lives in the
    # parts, and a campaign that measures V̂ from its own rollouts has no reason to read them.
    config, out_dir = campaign(tmp_path, parts_glob="/no/such/dir/*.jsonl",
                               splits_json="/no/such/dir/splits.json")
    values.build_values(config)
    assert written(out_dir)[0]["v_graded"] == pytest.approx(AT_BASELINE / 2)


def test_a_prefix_the_campaign_never_generated_for_is_counted_not_scored(tmp_path):
    config, out_dir = campaign(tmp_path, prefix_rows=[pre(), pre("p2")])
    manifest = values.build_values(config)
    assert len(written(out_dir)) == 1
    assert manifest["dropped"][values.NO_ROLLOUTS] == 1


def test_the_manifest_carries_the_whole_drop_ledger(tmp_path):
    config, out_dir = campaign(tmp_path, outcomes=(True, False, False))
    config.prm_rollout.min_rollouts = 4
    manifest = values.build_values(config)
    assert written(out_dir) == []
    assert manifest["dropped"][values.TOO_FEW_ROLLOUTS] == 1
    assert manifest["prefixes"] == 1
    assert manifest["rollouts"] == 3
    assert manifest["values"] == 0


def test_the_cli_scores_the_campaign_its_config_names(tmp_path):
    config, out_dir = campaign(tmp_path)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({"prm_rollout": dataclasses.asdict(config.prm_rollout)}))
    values.main(["--config", str(path)])
    assert (out_dir / values.VALUES_MANIFEST).exists()
    assert written(out_dir)[0]["prefix_id"] == "p1"
