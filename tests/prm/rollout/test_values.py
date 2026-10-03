"""``prm.rollout.values``: the eval results a campaign bought -> V̂ per prefix (PLAN_v2 §6, job D).

The invariant under test throughout is N1: every number here comes from the prefix's own
rollouts. v1's verdict for the completion the prefix was cut out of is one draw from a
different distribution, and nothing in this module may reach for it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
from collections import Counter

import pytest
import yaml

from processkernel.config import PRMRolloutConfig, RerankerConfig
from processkernel.prm.rollout import calibrate, orm_score, prefixes, rollout, stage, values

TAG = "ar"
LEVEL, PROBLEM = 2, 37
# speedup 1.0 against lo=0.2 hi=4.0 quant=0.1 -> speed_p 0.5 -> graded target 0.75.
BASELINES = {LEVEL: {PROBLEM: {"mean": 1.0, "min": 1.0}}}
AT_BASELINE = 0.75
# A bare "import torch\n" fails processkernel.checker.submission's S1.1 (no ModelNew) -- irrelevant on the
# measured path (aggregate_measured never reads `.code`), but roll()'s default has to be
# submission-clean for the imputed-path tests below that reuse it unmodified.
CLEAN_CODE = "import torch\nimport torch.nn as nn\n\nclass ModelNew(nn.Module):\n    def forward(self, x):\n        return x\n"


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
        cut_index=20,
        cut_kind="code",
        n_cuts_total=80,
        rel_depth=0.25,
        list_key=f"{TAG}:{LEVEL}:{PROBLEM}:20",
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
        code=CLEAN_CODE,
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
    return values.aggregate_measured(
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
    row = values.aggregate_measured(
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
    row = values.aggregate_measured(pre(), rows, joined, BASELINES, cfg(), Counter())
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
        values.aggregate_measured(pre(), rows, {}, BASELINES, cfg(), Counter())


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


# ============================================================================================
# The imputed path (PLAN_v3 §8): ORM scores -> V̂ via job D's curve.
# ============================================================================================


class _StubCurve:
    """A single-band linear curve: predict(x) = x * scale, band(x) = 0 always. Real enough
    for aggregate_imputed's arithmetic without depending on calibrate.fit_joint (N7 -- this
    module never fits, it only ever reads a curve, real or stubbed)."""

    def __init__(self, scale: float = 0.2, resid_var: float = 0.0):
        self.resid_var_by_band = [resid_var]
        self._scale = scale

    def predict(self, x: float) -> float:
        return x * self._scale

    def band(self, x: float) -> int:
        return 0


class _BandedCurve:
    """Two bands split at x = 0, with DISTINCT residual variances -- the shape `_StubCurve`
    cannot express and the only shape that can tell `band(score + c)` from `band(score)`.
    `predict` stays linear so v_graded is unaffected either way: the whole difference lands in
    `se_imputed`, which is where dropping the offset from the band lookup would hide.
    """

    def __init__(self, lo_var: float = 0.01, hi_var: float = 0.09, scale: float = 0.1):
        self.resid_var_by_band = [lo_var, hi_var]
        self._scale = scale

    def predict(self, x: float) -> float:
        return x * self._scale

    def band(self, x: float) -> int:
        return 1 if x >= 0.0 else 0


S1_FAIL_CODE = "def (:\n"   # does not even parse -- processkernel.checker.submission's own S1.0 fixture
OK_CODE = CLEAN_CODE
# Loadable (compiles, has ModelNew.forward) but a real F1.2 "dead kernel" finding: `k` is
# defined and never launched, computed in torch instead. Lifted verbatim from
# tests/checker/submission/test_submission_analyzer.py's own CHEATING fixture, which that
# suite uses to prove the exact same thing: SubmissionAnalyzer has nothing to say about it.
F1_FAIL_CODE = (
    "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n\n"
    "@triton.jit\n"
    "def k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):\n"
    "    pass\n\n\n"
    "class ModelNew(nn.Module):\n"
    "    def forward(self, x):\n"
    "        return torch.conv2d(x, x)\n"
)


def _impute(scores, resid_var=0.0, offset=0.0, truncated=(), min_rollouts=1,
            s1_fail=(), f1_fail=(), prefix=None, config=None, curve=None):
    """``aggregate_imputed`` over synthetic rollouts, one per element of ``scores``."""
    p = prefix or pre()
    rows, score_map = [], {}
    for i, s in enumerate(scores):
        rid = f"p1__j{i:02d}"
        if i in s1_fail:
            code = S1_FAIL_CODE
        elif i in f1_fail:
            code = F1_FAIL_CODE
        else:
            code = OK_CODE
        rows.append(roll(rid, code=code, truncation="truncated" if i in truncated else "ok"))
        score_map[rid] = s
    offsets = {f"{p.level}:{p.problem_id}": {"c": offset}}
    return values.aggregate_imputed(
        p, rows, score_map, curve or _StubCurve(resid_var=resid_var), offsets,
        config or cfg(min_rollouts=min_rollouts), Counter()
    )


def _measure(targets, config=None):
    """``aggregate_measured`` over synthetic rollouts: any ``t > 0`` is a correct kernel with
    a speedup so large it saturates the graded ladder at 1.0 (`t == 0.0` incorrect). Exact
    intermediate values are not the point here -- only that both paths reaggregate the same
    way and that the imputed-only fields stay unset on a measured row.
    """
    conf = config or cfg()
    rows, joined = [], {}
    for i, t in enumerate(targets):
        rid = f"p1__j{i:02d}"
        rows.append(roll(rid))
        joined[rid] = values.Eval(verdict(correct=t > 0, runtime=0.001, fastest=0.001))
    return values.aggregate_measured(pre(), rows, joined, BASELINES, conf, Counter())


def _run_job_d(curve_scale: float) -> dict:
    """Three prefixes, aggregated both ways under one curve scale -- proof that the two
    paths are actually independent, not just documented as such."""
    out = {"measured": [], "imputed": []}
    for i in range(3):
        pid = f"p{i}"
        rows, joined = graded(True, False)
        mv = values.aggregate_measured(pre(pid), rows, joined, BASELINES, cfg(), Counter())
        out["measured"].append(mv.v_graded)

        irows = [roll(f"{pid}__j00"), roll(f"{pid}__j01")]
        iscores = {f"{pid}__j00": 1.0 + i, f"{pid}__j01": 2.0 + i}
        curve = _StubCurve(scale=curve_scale)
        iv = values.aggregate_imputed(pre(pid), irows, iscores, curve, {}, cfg(), Counter())
        out["imputed"].append(iv.v_graded)
    return out


def test_imputed_row_has_null_counts_not_zero():
    # 0 would make measured and imputed silently averageable, which N1 forbids.
    v = _impute(scores=[1.0, 2.0])
    assert v.n_correct is None and v.n_compiled is None
    assert v.label_source == "imputed"


def test_measured_row_never_consults_the_orm():
    v = _measure(targets=[0.0, 1.0, 0.0])
    assert v.label_source == "measured"
    assert v.orm_offset is None and v.se_imputed is None


def test_se_imputed_matches_the_hand_computed_value():
    v = _impute(scores=[1.0, 3.0], resid_var=0.04)   # t̂ = 0.2, 0.6 on the stub curve
    want = math.sqrt(statistics.pvariance([0.2, 0.6]) / 2 + 0.04)
    assert v.se_imputed == pytest.approx(want, abs=1e-9)


def test_offset_is_applied_inside_the_lookup():
    # knots_x are already in corrected units, so applying c after would double-count it.
    hot = _impute(scores=[1.0], offset=+2.0).v_graded
    cold = _impute(scores=[1.0], offset=-2.0).v_graded
    assert hot > cold
    assert _impute(scores=[3.0], offset=0.0).v_graded == pytest.approx(hot)


def test_the_band_is_looked_up_at_the_offset_corrected_score_too():
    # `x = score + c` feeds predict AND band. Dropping `+ c` from the band lookup alone leaves
    # every v_graded identical and silently shifts every se_imputed in the campaign, which is
    # why this needs two bands with different residual variances to be visible at all.
    # score -1.0 with c = +2.0 is x = +1.0: the HIGH band. Banding the raw -1.0 reads the low.
    v = _impute(scores=[-1.0, -1.0], offset=+2.0, curve=_BandedCurve(0.01, 0.09))
    assert v.se_imputed == pytest.approx(math.sqrt(0.09))    # not sqrt(0.01)


def test_se_imputed_averages_each_rollouts_own_bands_residual_variance():
    # Straddling the split: one rollout per band, so the residual term is the mean of the two
    # and not whichever band the first rollout happened to land in.
    v = _impute(scores=[-1.0, 1.0], curve=_BandedCurve(0.01, 0.09))
    want = math.sqrt(statistics.pvariance([-0.1, 0.1]) / 2 + (0.01 + 0.09) / 2)
    assert v.se_imputed == pytest.approx(want, abs=1e-12)


def test_truncated_rollout_is_dropped_before_imputation():
    v = _impute(scores=[1.0, 2.0, 3.0], truncated=[2])
    assert v.n_rollouts == 2


def test_min_rollouts_fires_on_the_imputed_path_too():
    assert _impute(scores=[1.0], min_rollouts=3) is None


def test_min_rollouts_is_counted_after_the_drops_not_before():
    # 3 rollouts arrived, 2 were truncated: the prefix has ONE usable draw against a floor of
    # 2 and must leave the corpus. Counting arrivals instead would keep a V-hat from one draw.
    assert _impute(scores=[1.0, 2.0, 3.0], truncated=[1, 2], min_rollouts=2) is None
    assert _impute(scores=[1.0, 2.0, 3.0], truncated=[2], min_rollouts=2).n_rollouts == 2


def test_an_imputed_row_leaves_every_measured_only_field_null():
    # N1's other three: v_binary/se_binary/se_graded are verdict-derived, and a 0.0 there would
    # read as "measured, and it was zero" -- averageable with a real measured row.
    v = _impute(scores=[1.0, 2.0])
    assert v.v_binary is None and v.se_binary is None and v.se_graded is None


def test_changing_the_curve_moves_every_shell_label_and_no_core_label():
    a = _run_job_d(curve_scale=1.0)
    b = _run_job_d(curve_scale=0.5)
    assert all(x != y for x, y in zip(a["imputed"], b["imputed"]))
    assert a["measured"] == b["measured"]


def test_scores_reaggregate_to_v_graded_on_both_paths():
    for v in (_measure(targets=[0.0, 1.0]), _impute(scores=[1.0, 3.0])):
        assert sum(v.scores) / len(v.scores) == pytest.approx(v.v_graded)


def test_an_s1_failure_scores_zero_without_touching_the_curve():
    # 0 of 25 such kernels were correct and the evaluator cannot even import the file, so
    # this is a deduction. A curve that would map its score high must not get the chance.
    v = _impute(scores=[9.0, 1.0], s1_fail=[0])
    assert v.scores[0] == 0.0
    assert v.n_rollouts == 2


def test_a_linter_hard_fail_is_still_imputed_normally():
    # Regression guard: F1 covers 36% of kernels and holds 28% of all correct ones. A future
    # "gate on the linter too" refactor must fail here, not in production. The premise is
    # checked, not assumed: F1_FAIL_CODE really does trip a real F1 finding today.
    from processkernel import checker

    assert any(f.check_id.startswith("F1.") for f in checker.analyze_source(F1_FAIL_CODE).findings)
    v = _impute(scores=[9.0], f1_fail=[0], min_rollouts=1)
    assert v.scores[0] > 0.0


# --- five requirements from the reviews of Tasks 3-5, not in the brief but load-bearing ----


def test_submission_ok_matches_checker_submission_directly():
    assert values.submission_ok(OK_CODE) is True
    assert values.submission_ok(S1_FAIL_CODE) is False


# (a) campaign completeness: every landed rollout unit must have a score part -------------


def test_assert_scores_complete_names_the_unscored_unit():
    by_unit = {"u0": [roll("p1__j00")], "u1": [roll("p2__j00")]}
    with pytest.raises(ValueError, match="u1"):
        values._assert_scores_complete(["u0", "u1"], by_unit, {"p1__j00": {}})


def test_assert_scores_complete_names_a_unit_whose_rollout_part_never_landed():
    # The QoS cap runs a 16-task array in waves, so a dead task is a live failure mode: u1
    # is in the plan and left nothing on disk. An expected set globbed off the parts cannot
    # see it -- by_unit has no u1 key to be wrong about -- and the campaign writes short.
    by_unit = {"u0": [roll("p1__j00")]}
    with pytest.raises(ValueError, match="u1"):
        values._assert_scores_complete(["u0", "u1"], by_unit, {"p1__j00": {}})


def test_assert_scores_complete_passes_when_every_planned_unit_landed_and_scored():
    by_unit = {"u0": [roll("p1__j00")], "u1": [roll("p2__j00")]}
    values._assert_scores_complete(
        ["u0", "u1"], by_unit, {"p1__j00": {}, "p2__j00": {}}
    )   # no raise


# (b) label_source gate: calibrate.py no longer refuses a measured fit, so this is the only
# barrier left against labelling a real campaign from a curve fit for validation only --------


def test_read_calib_manifest_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        values._read_calib_manifest(str(tmp_path))


def test_read_calib_manifest_refuses_a_measured_fit(tmp_path):
    (tmp_path / calibrate.CALIB_MANIFEST).write_text(
        json.dumps({"label_source": "measured", "anchors": {}})
    )
    with pytest.raises(ValueError, match="label_source"):
        values._read_calib_manifest(str(tmp_path))


def test_read_calib_manifest_accepts_an_imputed_fit(tmp_path):
    (tmp_path / calibrate.CALIB_MANIFEST).write_text(
        json.dumps({"label_source": "imputed", "anchors": {}})
    )
    assert values._read_calib_manifest(str(tmp_path))["label_source"] == "imputed"


# (c), (d) curve.json is validated on load, not consumed blindly -----------------------------


def _curve_json(**over):
    d = {"knots_x": [0.0, 5.0], "knots_y": [0.0, 1.0], "resid_var_by_band": [0.01, 0.01],
         "n_by_band": [2, 2], "n_fit": 4}
    d.update(over)
    return d


def test_load_curve_refuses_non_ascending_knots(tmp_path):
    (tmp_path / calibrate.CURVE).write_text(json.dumps(_curve_json(knots_x=[5.0, 0.0])))
    with pytest.raises(ValueError, match="ascending"):
        values.load_curve(str(tmp_path))


def test_load_curve_refuses_equal_adjacent_knots(tmp_path):
    # np.interp tolerates a tie, so nothing downstream would raise -- but band()'s
    # searchsorted(side="right") can then never land on the earlier of the two, so that band's
    # resid_var_by_band entry is dead and every se_imputed there reports its neighbour's.
    (tmp_path / calibrate.CURVE).write_text(json.dumps(_curve_json(knots_x=[0.0, 0.0])))
    with pytest.raises(ValueError, match="ascending"):
        values.load_curve(str(tmp_path))


def test_load_curve_refuses_an_all_zero_resid_var(tmp_path):
    (tmp_path / calibrate.CURVE).write_text(
        json.dumps(_curve_json(resid_var_by_band=[0.0, 0.0]))
    )
    with pytest.raises(ValueError, match="resid_var_by_band"):
        values.load_curve(str(tmp_path))


def test_load_curve_accepts_a_healthy_curve(tmp_path):
    (tmp_path / calibrate.CURVE).write_text(json.dumps(_curve_json()))
    curve = values.load_curve(str(tmp_path))
    assert curve.predict(0.0) == pytest.approx(0.0)


# --- the pass end to end: an on-disk imputed campaign ---------------------------------------


def _orm_row(rid, score=1.0, ckpt="ck1", n_tokens=50) -> dict:
    return {"kind": "rollout", "id": rid, "level": LEVEL, "problem_id": PROBLEM,
            "code_sha1": "a" * 40, "orm_score": score, "orm_checkpoint_sha": ckpt,
            "n_code_tokens": n_tokens}


def _offsets_json(offsets=None) -> dict:
    meta = {"tau2": 0.1, "tau2_raw": 0.1, "tau2_estimable": True, "kappa_mode": "auto",
            "c_std": 0.0, "c_std_core": 0.0, "n_clamped": 0, "n_no_anchors": 0,
            "n_no_info": 0, "n_informative": 1, "frac_informative": 1.0,
            "shrink_mean": 1.0, "n_problems": 1}
    return {"meta": meta, "offsets": offsets or {}}


def _calib_manifest_json(label_source="imputed", ckpt="ck1") -> dict:
    return {"created": "T", "config": {}, "label_source": label_source,
            "anchors": {"orm_checkpoint_sha": ckpt}, "fit": {}, "curve": {},
            "curve_sha1": "cs1", "offsets_sha1": "os1"}


# The unit names `rollout.unit_name` gives this file's prefixes: run__shard__roundN. Not
# arbitrary strings any more -- values.py now takes its expected unit set from prefixes.jsonl
# (via orm_score.rollout_units), so a fixture whose parts are named u0/u1 describes a campaign
# where every planned unit is missing and every landed one unplanned.
U0, U1 = "a_run__shard_00", "a_run__shard_01"


def imputed_campaign(tmp_path, *, unit_rids: dict, scored_units, score_ckpt="ck1",
                     manifest_ckpt="ck1", curve=None, offsets=None, score=1.0,
                     calib_label_source="imputed", prefix_rows=None, land_units=None):
    """Every artifact jobs A, B, B2 and D write for an imputed campaign, laid out as they lay
    it out: ``unit_rids`` maps unit name -> the rollout_ids landed under it, ``scored_units``
    is the subset that also gets an ``orm_scores.jsonl/<unit>.jsonl`` part, and ``land_units``
    (default: all of them) is the subset whose rollout part is written at all -- a unit outside
    it is one whose array task died, planned in ``prefixes.jsonl`` and absent from disk.
    """
    out_dir = tmp_path / "campaign"
    (out_dir / stage.ROLLOUTS).mkdir(parents=True)
    (out_dir / orm_score.ORM_SCORES).mkdir(parents=True)
    for unit, rids in unit_rids.items():
        if land_units is None or unit in land_units:
            stage.write_rollouts(
                [roll(rid) for rid in rids], str(out_dir / stage.ROLLOUTS / f"{unit}.jsonl.gz")
            )
        if unit in scored_units:
            lines = "".join(
                json.dumps(_orm_row(rid, score=score, ckpt=score_ckpt)) + "\n" for rid in rids
            )
            (out_dir / orm_score.ORM_SCORES / f"{unit}.jsonl").write_text(lines)
    (out_dir / prefixes.PREFIXES).write_text(
        "".join(json.dumps(dataclasses.asdict(p)) + "\n" for p in (prefix_rows or [pre()]))
    )
    (out_dir / calibrate.CURVE).write_text(json.dumps(curve if curve is not None else _curve_json()))
    (out_dir / calibrate.OFFSETS).write_text(
        json.dumps(offsets if offsets is not None else _offsets_json())
    )
    (out_dir / calibrate.CALIB_MANIFEST).write_text(
        json.dumps(_calib_manifest_json(label_source=calib_label_source, ckpt=manifest_ckpt))
    )
    config = RerankerConfig(prm_rollout=cfg(
        out_dir=str(out_dir), label_source="imputed", orm_checkpoint="ck", min_rollouts=1,
    ))
    return config, out_dir


def test_build_values_imputed_writes_rows_end_to_end(tmp_path):
    config, out_dir = imputed_campaign(
        tmp_path, unit_rids={U0: ["p1__j00", "p1__j01"]}, scored_units={U0}
    )
    manifest = values.build_values(config)
    assert manifest["label_source"] == "imputed"
    assert manifest["values"] == 1
    (row,) = written(out_dir)
    assert row["label_source"] == "imputed"
    assert row["n_correct"] is None and row["n_compiled"] is None
    assert row["se_imputed"] is not None


def test_build_values_dispatches_to_the_imputed_path(tmp_path):
    # build_values, not build_values_imputed directly: the dispatcher is what main() and every
    # existing measured-path caller actually run.
    config, out_dir = imputed_campaign(
        tmp_path, unit_rids={U0: ["p1__j00", "p1__j01"]}, scored_units={U0}
    )
    values.build_values(config)
    assert (out_dir / values.VALUES).exists()


def test_build_values_imputed_refuses_when_a_unit_is_unscored(tmp_path):
    config, _ = imputed_campaign(
        tmp_path,
        unit_rids={U0: ["p1__j00"], U1: ["p2__j00"]},
        scored_units={U0},
        prefix_rows=[pre("p1"), pre("p2", shard="shard_01")],
    )
    with pytest.raises(ValueError, match="shard_01"):
        values.build_values(config)


def test_build_values_imputed_refuses_when_a_planned_units_part_never_landed(tmp_path):
    # The dead-array-task case, end to end: prefixes.jsonl plans two units, only one landed.
    # Globbing the parts for the expected set passes this campaign and writes p2's prefixes
    # off as no_rollouts -- a 1.1M-row campaign short by a sixteenth, with no error anywhere.
    config, _ = imputed_campaign(
        tmp_path,
        unit_rids={U0: ["p1__j00"], U1: ["p2__j00"]},
        scored_units={U0},
        land_units={U0},
        prefix_rows=[pre("p1"), pre("p2", shard="shard_01")],
    )
    with pytest.raises(ValueError, match="shard_01"):
        values.build_values(config)


def test_build_values_imputed_refuses_a_measured_calibrate_manifest(tmp_path):
    config, _ = imputed_campaign(
        tmp_path, unit_rids={U0: ["p1__j00"]}, scored_units={U0},
        calib_label_source="measured",
    )
    with pytest.raises(ValueError, match="label_source"):
        values.build_values(config)


def test_build_values_imputed_refuses_a_checkpoint_mismatch(tmp_path):
    config, _ = imputed_campaign(
        tmp_path, unit_rids={U0: ["p1__j00"]}, scored_units={U0},
        score_ckpt="wrong-ckpt", manifest_ckpt="ck1",
    )
    with pytest.raises(ValueError, match="orm_checkpoint_sha"):
        values.build_values(config)


def test_the_offset_on_disk_reaches_the_lookup_end_to_end(tmp_path):
    # The real read_offsets -> offset_for path, not a hand-built dict: _offsets_json defaults
    # to no offsets at all, so every end-to-end test above runs at c = 0 and the whole
    # per-problem correction could be dropped without one of them noticing.
    # score 1.0 on knots (0, 5) -> (0, 1) is 0.2; with c = +1.0 the lookup is at 2.0 -> 0.4.
    def run(offsets):
        config, out_dir = imputed_campaign(
            tmp_path / str(bool(offsets)), unit_rids={U0: ["p1__j00"]}, scored_units={U0},
            offsets=_offsets_json(offsets),
        )
        values.build_values(config)
        return written(out_dir)[0]

    plain, shifted = run({}), run({f"{LEVEL}:{PROBLEM}": {"c": 1.0}})
    assert plain["orm_offset"] == 0.0 and plain["v_graded"] == pytest.approx(0.2)
    assert shifted["orm_offset"] == 1.0 and shifted["v_graded"] == pytest.approx(0.4)


def test_build_values_imputed_reports_the_campaigns_clip_rate(tmp_path):
    # curve.n_clipped is a live counter aggregate_imputed drives and used to discard. A score
    # of 9.0 is past the curve's last knot (5.0), so the label is the flat extrapolation --
    # the distribution-shift signal, and free to count.
    config, _ = imputed_campaign(
        tmp_path, unit_rids={U0: ["p1__j00", "p1__j01"]}, scored_units={U0}, score=9.0
    )
    manifest = values.build_values(config)
    assert manifest["curve_lookups"] == 2
    assert manifest["clipped"] == 2
    assert manifest["clip_rate_pct"] == pytest.approx(100.0)


def test_a_campaign_inside_the_curves_range_clips_nothing(tmp_path):
    config, _ = imputed_campaign(
        tmp_path, unit_rids={U0: ["p1__j00"]}, scored_units={U0}, score=1.0
    )
    manifest = values.build_values(config)
    assert manifest["clipped"] == 0 and manifest["clip_rate_pct"] == 0.0
