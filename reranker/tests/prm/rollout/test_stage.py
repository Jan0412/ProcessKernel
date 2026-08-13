"""``prm.rollout.stage``: rollouts -> kernels the eval harness can read (PLAN_v2 §6, job C's input).

The harness is the constraint this module is shaped by, so the tests are written against how it
actually behaves rather than how §6 describes it: it takes one `level`, narrows problems by a
contiguous `subset` range, and walks `range(num_samples_per_problem)` for every problem in that
range whether or not a file is there.
"""

from __future__ import annotations

import dataclasses
import glob
import json

import pytest
import yaml

from reranker.src.config import PRMRolloutConfig, RerankerConfig
from reranker.src.data.build_dataset import _staged_kernel_path
from reranker.src.prm.rollout import prefixes, rollout, stage

TAG = "ar"


def _kernels(run_dir) -> list[str]:
    return sorted(p.name for p in run_dir.iterdir())


def roll(rid="r0", sha="a" * 40, code="import torch\n", **over):
    fields = dict(
        rollout_id=rid,
        prefix_id=rid.rsplit("__j", 1)[0],
        j=0,
        continuation="...",
        code=code,
        code_sha1=sha,
        n_prefix_tokens=10,
        n_gen_tokens=20,
        finish_reason={"plan": None, "code": "stop"},
        truncation="ok",
    )
    fields.update(over)
    return rollout.Rollout(**fields)


def where(*triples) -> dict:
    """``prefix_id -> (level, problem_id)``, which the rollout row itself does not carry."""
    return {pid: (level, problem) for pid, level, problem in triples}


def pre(prefix_id="p1", *, level=6, problem_id=1, **over):
    fields = dict(
        prefix_id=prefix_id,
        source="cut",
        run_name="a_run",
        run_tag=TAG,
        shard="shard_00",
        round=0,
        level=level,
        problem_id=problem_id,
        sample_id=0,
        stem=f"level_{level}_problem_{problem_id}_sample_0_kernel",
        cut_char=10,
        cut_index=5,
        cut_kind="code",
        n_cuts_total=20,
        rel_depth=0.25,
        list_key=f"{TAG}:{level}:{problem_id}:0:5",
        split="train",
        selection="random",
        selection_score=None,
        K=2,
        min_rollouts=1,
    )
    fields.update(over)
    return prefixes.Prefix(**fields)


def cfg(**over) -> PRMRolloutConfig:
    base = PRMRolloutConfig(run_tags={"a_run": TAG}, baseline_timing_json=__file__, min_rollouts=1)
    for k, v in over.items():
        setattr(base, k, v)
    base.validate()
    return base


def campaign(per_problem: dict[int, int], level=6):
    """``({problem_id: how many}) -> (staged rollouts, where)``, each with a body of its own."""
    rows, w = [], {}
    for problem, n in per_problem.items():
        for j in range(n):
            pid = f"p{problem}s{j}"
            rows.append(roll(f"{pid}__j00", sha=f"{problem}:{j}"))
            w[pid] = (level, problem)
    return stage.assign(rows, w), w


def assigned(per_problem: dict[int, int], level=6):
    return campaign(per_problem, level)[0]


def test_two_rollouts_of_one_problem_with_the_same_code_are_staged_once():
    rows = [roll("p1__j00", sha="dup"), roll("p1__j01", sha="dup")]
    staged = stage.assign(rows, where(("p1", 6, 1)))
    assert staged[0].staged_as == {"level": 6, "problem_id": 1, "sample_id": 0}
    assert staged[1].staged_as is None


def test_the_same_code_under_two_problems_is_staged_twice():
    # Eval runs a kernel against ITS problem's reference architecture, so identical bytes under
    # two problems are two different measurements. extract_code_block returns "" for a rollout
    # that emitted no fence, so every empty body in a campaign shares one sha -- deduping on the
    # sha alone would resolve problem 2's failures through problem 1's eval result.
    rows = [roll("p1__j00", sha="dup"), roll("p2__j00", sha="dup")]
    staged = stage.assign(rows, where(("p1", 6, 1), ("p2", 6, 2)))
    assert staged[0].staged_as == {"level": 6, "problem_id": 1, "sample_id": 0}
    assert staged[1].staged_as == {"level": 6, "problem_id": 2, "sample_id": 0}


def test_sample_ids_are_dense_from_zero_within_each_problem():
    # The harness walks range(num_samples_per_problem); every id it skips is an eval slot paid
    # for and thrown away, so a sparse assignment costs real GPU time.
    rows = [roll(f"p1__j{j:02d}", sha=f"s{j}") for j in range(4)]
    staged = stage.assign(rows, where(("p1", 6, 1)))
    assert [r.staged_as["sample_id"] for r in staged] == [0, 1, 2, 3]


def test_the_assignment_does_not_depend_on_the_order_the_rollouts_arrive_in():
    # generate() returns rollouts in budget-bucket order, which changes with the batch. If that
    # reached the sample ids, a rerun would stage the same kernel under a different id and the
    # map would stop matching an eval_results.json written by the previous run.
    rows = [roll(f"p1__j{j:02d}", sha=f"s{j}") for j in range(4)]
    w = where(("p1", 6, 1))
    forward = {r.rollout_id: r.staged_as for r in stage.assign(rows, w)}
    backward = {r.rollout_id: r.staged_as for r in stage.assign(rows[::-1], w)}
    assert forward == backward


def test_a_rollout_whose_prefix_is_not_in_the_campaign_raises():
    with pytest.raises(KeyError, match="p9__j00"):
        stage.assign([roll("p9__j00")], where(("p1", 6, 1)))


def test_two_rollouts_sharing_an_id_raise_rather_than_collapsing_onto_one_slot():
    # Measured: placement is keyed by rollout_id, so the second row overwrites the first's
    # entry and both end up staged as sample 1. Sample 0 is never written, write_kernels
    # os.replaces one body over the other, and both read one eval result.
    rows = [roll("p1__j00", sha="aaa"), roll("p1__j00", sha="bbb")]
    with pytest.raises(ValueError, match="p1__j00"):
        stage.assign(rows, where(("p1", 6, 1)))


def test_dedup_can_be_turned_off_and_then_every_rollout_gets_its_own_slot():
    # §7's dedup_by_code_sha1. Off, the campaign pays for duplicate evals; it is the control
    # arm for "is the dedup resolving rollouts to the wrong result?", which is otherwise
    # unanswerable from the campaign's own output.
    rows = [roll("p1__j00", sha="dup"), roll("p1__j01", sha="dup")]
    staged = stage.assign(rows, where(("p1", 6, 1)), dedup=False)
    assert [r.staged_as["sample_id"] for r in staged] == [0, 1]


def test_homes_reads_level_and_problem_off_the_prefixes():
    ps = [pre("p1", problem_id=1), pre("p2", problem_id=2)]
    assert stage.homes(ps) == {"p1": (6, 1), "p2": (6, 2)}


def test_two_prefixes_sharing_an_id_raise_rather_than_shadowing_each_other():
    # run_tags may map two run_names onto one tag, which nothing upstream rejects. A dict would
    # keep the last of the pair, and the shadowed prefix's rollouts would be graded against
    # another problem's reference architecture with no error anywhere.
    with pytest.raises(ValueError, match="p1"):
        stage.homes([pre("p1", problem_id=1), pre("p1", problem_id=2)])


# --- shards: the harness narrows by a contiguous range, so the partition has to be one ----


def test_each_shard_gets_its_own_run_dir_and_a_range_that_covers_its_problems():
    # One run dir per shard is not a preference: add_to_eval_results_file appends by
    # load-rewrite of the whole eval_results.json, so two jobs sharing one file drop each
    # other's results.
    staged = assigned({1: 2, 2: 2, 3: 2, 4: 2})
    shards = stage.plan_shards(staged, cfg(eval_shards=2))
    assert [s.run_name for s in shards] == ["prm_rollout_v1_s00", "prm_rollout_v1_s01"]
    assert [s.subset for s in shards] == [(1, 2), (3, 4)]
    assert [s.problems for s in shards] == [(1, 2), (3, 4)]


def test_a_shard_asks_for_as_many_samples_as_its_busiest_problem_has():
    # num_samples_per_problem is one number for the whole shard and the harness walks
    # range() of it: below the max, the tail of the busiest problem is never evaluated at all.
    staged = assigned({1: 3, 2: 7})
    (shard,) = stage.plan_shards(staged, cfg(eval_shards=1))
    assert shard.num_samples_per_problem == 7


def test_shards_balance_the_staged_kernels_rather_than_the_problem_count():
    staged = assigned({1: 10, 2: 1, 3: 1, 4: 1})
    shards = stage.plan_shards(staged, cfg(eval_shards=2))
    assert [s.problems for s in shards] == [(1,), (2, 3, 4)]


@pytest.mark.parametrize("shards", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "counts",
    [
        {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},      # flat
        {1: 34, 2: 34, 3: 34},                      # every problem exactly at the target
        {1: 100, 2: 1, 3: 1},                       # one problem carries the campaign
        {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},             # rising
    ],
)
def test_the_partition_never_emits_more_shards_than_the_array_was_sized_for(counts, shards):
    # The loop closes a group whenever it reaches its share, with nothing counting the groups.
    # That stays within eval_shards only because a closed group holds at least total/eval_shards,
    # so that many closes spend the whole total and the flush has nothing left -- an argument
    # worth a test rather than a comment, since an extra shard is one the sbatch array never runs.
    got = stage.plan_shards(assigned(counts), cfg(eval_shards=shards))
    assert 1 <= len(got) <= shards
    assert [p for s in got for p in s.problems] == sorted(counts)


def test_a_shard_with_no_kernels_is_not_emitted():
    # The sbatch array is sized off the shard count; an empty shard would be a job whose
    # subset selects problems it staged nothing for, evaluating gaps for its whole wall clock.
    staged = assigned({1: 2})
    shards = stage.plan_shards(staged, cfg(eval_shards=4))
    assert [s.run_name for s in shards] == ["prm_rollout_v1_s00"]


def test_deduped_rollouts_do_not_take_up_a_slot_in_the_shard_plan():
    rows = stage.assign(
        [roll("p1__j00", sha="dup"), roll("p1__j01", sha="dup")], where(("p1", 6, 1))
    )
    (shard,) = stage.plan_shards(rows, cfg(eval_shards=1))
    assert shard.num_samples_per_problem == 1 and shard.n_staged == 1


# --- the map job D joins back through -------------------------------------------------


def test_a_deduped_rollout_resolves_through_the_kernel_that_was_actually_staged():
    # The whole point of the dedup: the second rollout has no eval of its own, and its V̂
    # contribution is the first one's result. Without this it would be dropped as no_eval_entry
    # and the dedup would quietly cost measurements instead of saving evals.
    w = where(("p1", 6, 1))
    rows = stage.assign([roll("p1__j00", sha="dup"), roll("p1__j01", sha="dup")], w)
    m = stage.rollout_map(rows, w)
    assert m["p1__j00"] == {"level": 6, "problem_id": 1, "sample_id": 0, "shared": False}
    assert m["p1__j01"] == {"level": 6, "problem_id": 1, "sample_id": 0, "shared": True}


def test_the_map_holds_every_rollout_and_recovers_the_rows_it_came_from():
    # §5: the map is an index derived from the rows, not a second source of truth. One assign
    # pass, with the duplicate body inside it -- a second pass restarts its sample counter, so
    # a row appended that way is not deduped at all and the round trip below never reaches the
    # branch this is about.
    w = where(("p1s0", 6, 1), ("p1s1", 6, 1), ("p2s0", 6, 2))
    rows = stage.assign(
        [
            roll("p1s0__j00", sha="1:0"),
            roll("p1s0__j01", sha="1:0"),
            roll("p1s1__j00", sha="1:1"),
            roll("p2s0__j00", sha="2:0"),
        ],
        w,
    )
    assert sum(r.staged_as is None for r in rows) == 1
    m = stage.rollout_map(rows, w)
    assert set(m) == {r.rollout_id for r in rows}
    for r in rows:
        entry = dict(m[r.rollout_id])
        assert r.staged_as == (None if entry.pop("shared") else entry)


def test_the_map_does_not_depend_on_row_order():
    rows, w = campaign({1: 3, 2: 2})
    assert stage.rollout_map(rows, w) == stage.rollout_map(rows[::-1], w)


def test_a_deduped_rollout_with_no_staged_original_raises():
    # Only reachable by staging a slice of the campaign's rows, which is exactly the bug: the
    # owner sits in another part, and the map would silently point the rollout at nothing.
    with pytest.raises(ValueError, match="p1__j01"):
        stage.rollout_map([roll("p1__j01", sha="dup")], where(("p1", 6, 1)))


def test_the_same_body_under_two_problems_resolves_to_its_own_problems_kernel():
    # The empty-code case, which is what makes this more than an accounting detail: an
    # unfenced rollout under problem 1 and another under problem 2 share a sha, and resolving
    # problem 2's through problem 1's eval would report a result measured on another problem.
    w = where(("p1", 6, 1), ("p2", 6, 2))
    rows = stage.assign(
        [roll("p1__j00", sha="dup"), roll("p1__j01", sha="dup"), roll("p2__j00", sha="dup")], w
    )
    m = stage.rollout_map(rows, w)
    assert m["p1__j01"]["problem_id"] == 1 and m["p1__j01"]["shared"] is True
    assert m["p2__j00"]["problem_id"] == 2 and m["p2__j00"]["shared"] is False


# --- what lands on disk ----------------------------------------------------------------


def test_rollout_rows_survive_the_round_trip_through_the_gzipped_part(tmp_path):
    rows = assigned({1: 2})
    path = str(tmp_path / "u.jsonl.gz")
    stage.write_rollouts(rows, path)
    assert stage.read_rollouts(path) == rows


def test_the_part_is_written_by_rename_and_leaves_no_half_file_behind(tmp_path):
    # §9's idempotence rests on this: a part that exists means a unit that finished, so a job
    # killed mid-write must not leave something a rerun would mistake for done.
    stage.write_rollouts(assigned({1: 1}), str(tmp_path / "u.jsonl.gz"))
    assert [p.name for p in tmp_path.iterdir()] == ["u.jsonl.gz"]


def test_a_staged_kernel_lands_where_the_dataset_reader_looks_for_it(tmp_path):
    # Pinned against the reader rather than restated: build_dataset.py joins eval results back
    # to sources through this exact name, and a second spelling of it here would drift.
    rows = stage.assign([roll("p1__j00", code="import torch\n")], where(("p1", 6, 1)))
    stage.write_kernels(rows, stage.plan_shards(rows, cfg())[0], str(tmp_path))
    path = _staged_kernel_path(str(tmp_path / "prm_rollout_v1_s00"), 6, 1, 0)
    assert open(path).read() == "import torch\n"


def test_a_kernel_body_holding_a_lone_surrogate_is_staged_rather_than_aborting(tmp_path):
    # `code` is sliced out of raw, which v1 writes ensure_ascii, so a half-character in the
    # corpus decodes straight back into it. The one write here that is not ASCII JSON, and
    # aborting on it kills the campaign's staging pass with some shards already on disk.
    rows = stage.assign([roll("p1__j00", code="x = 'a\ud800b'\n")], where(("p1", 6, 1)))
    assert stage.write_kernels(rows, stage.plan_shards(rows, cfg())[0], str(tmp_path)) == 1


def test_a_deduped_rollout_writes_no_kernel_file(tmp_path):
    w = where(("p1", 6, 1))
    rows = stage.assign([roll("p1__j00", sha="dup"), roll("p1__j01", sha="dup")], w)
    assert stage.write_kernels(rows, stage.plan_shards(rows, cfg())[0], str(tmp_path)) == 1


def test_a_shard_writes_only_its_own_problems(tmp_path):
    # Every kernel in the wrong dir is an eval another shard already paid for, and its result
    # lands in a second eval_results.json under the same (problem_id, sample_id) key.
    rows = assigned({1: 1, 2: 1})
    first, second = stage.plan_shards(rows, cfg(eval_shards=2))
    stage.write_kernels(rows, first, str(tmp_path))
    stage.write_kernels(rows, second, str(tmp_path))
    assert _kernels(tmp_path / first.run_name) == ["level_6_problem_1_sample_0_kernel.py"]
    assert _kernels(tmp_path / second.run_name) == ["level_6_problem_2_sample_0_kernel.py"]


def test_staging_into_a_run_dir_that_already_holds_kernels_raises(tmp_path):
    # A re-stage assigns sample ids over whatever rollouts exist now, so last run's kernels
    # keep ids that mean something else -- and its eval_results.json grades them under the new
    # map. Refusing is the only safe move; clearing the dir is the operator's call.
    rows = assigned({1: 1})
    shard = stage.plan_shards(rows, cfg())[0]
    stage.write_kernels(rows, shard, str(tmp_path))
    with pytest.raises(FileExistsError, match=shard.run_name):
        stage.write_kernels(rows, shard, str(tmp_path))


def test_staging_into_a_run_dir_that_kept_its_eval_results_raises(tmp_path):
    # Worse than keeping both: the kernel guard passes, fresh ids are assigned, and the harness
    # skips every re-staged one as already evaluated. Job D then joins the new map to the
    # previous pass's results and nothing says so.
    rows = assigned({1: 1})
    shard = stage.plan_shards(rows, cfg())[0]
    run_dir = tmp_path / shard.run_name
    run_dir.mkdir(parents=True)
    (run_dir / "eval_results.json").write_text("{}")   # the name the harness writes
    with pytest.raises(FileExistsError, match="eval_results.json"):
        stage.write_kernels(rows, shard, str(tmp_path))


# --- the whole pass, over a campaign on disk --------------------------------------------


def written(tmp_path, per_problem=None, parts=1, **over):
    """A campaign as job A and the generation job leave it: prefixes.jsonl + gzipped parts."""
    out = tmp_path / "campaign"
    out.mkdir()
    rows, w = campaign({1: 2, 2: 2} if per_problem is None else per_problem)
    rows = [dataclasses.replace(r, staged_as=None) for r in rows]
    (out / prefixes.PREFIXES).write_text(
        "".join(
            json.dumps(dataclasses.asdict(pre(pid, problem_id=problem))) + "\n"
            for pid, (_, problem) in sorted(w.items())
        )
    )
    for i in range(parts):
        stage.write_rollouts(
            rows[i::parts], str(out / stage.ROLLOUTS / f"unit_{i}.jsonl.gz")
        )
    return RerankerConfig(
        prm_rollout=cfg(
            out_dir=str(out), eval_runs_dir=str(tmp_path / "runs"), **over
        )
    )


def test_the_pass_stages_every_kernel_and_leaves_the_map_beside_the_rollouts(tmp_path):
    conf = written(tmp_path, eval_shards=1)
    manifest = stage.stage_campaign(conf)
    run_dir = tmp_path / "runs" / "prm_rollout_v1_s00"
    assert _kernels(run_dir) == [
        "level_6_problem_1_sample_0_kernel.py", "level_6_problem_1_sample_1_kernel.py",
        "level_6_problem_2_sample_0_kernel.py", "level_6_problem_2_sample_1_kernel.py",
    ]
    m = json.loads((tmp_path / "campaign" / stage.ROLLOUT_MAP).read_text())
    assert len(m) == 4 and manifest["staged"] == 4


def test_the_staged_ids_are_written_back_into_the_rollout_rows(tmp_path):
    # §5 puts staged_as on the row and derives the map from it. If the pass only wrote the map,
    # the expensive artifact would be the one file that cannot say where its rollouts were graded.
    conf = written(tmp_path, parts=2)
    stage.stage_campaign(conf)
    parts = sorted(glob.glob(str(tmp_path / "campaign" / stage.ROLLOUTS / "*.jsonl.gz")))
    rows = [r for p in parts for r in stage.read_rollouts(p)]
    assert len(rows) == 4 and all(r.staged_as is not None for r in rows)


def test_the_manifest_carries_what_the_eval_jobs_have_to_be_launched_with(tmp_path):
    # prm_eval.sh reads these; a shard whose subset or sample count is not recorded here is a
    # shard nobody can submit.
    conf = written(tmp_path, per_problem={1: 2, 2: 2, 3: 2, 4: 2}, eval_shards=2)
    manifest = stage.stage_campaign(conf)
    assert [s["run_name"] for s in manifest["shards"]] == [
        "prm_rollout_v1_s00", "prm_rollout_v1_s01",
    ]
    assert manifest["shards"][0] == {
        "index": 0, "run_name": "prm_rollout_v1_s00", "level": 6, "subset": [1, 2],
        "problems": 2, "num_samples_per_problem": 2, "n_staged": 4, "n_work": 4,
    }


def test_the_manifest_carries_the_eval_knobs_so_the_wrapper_reads_one_source(tmp_path):
    # prm_eval.sh passes these straight through to eval_from_generations. Re-typing them into
    # the sbatch script is how a campaign ends up graded under different trial counts than the
    # config it claims to have run.
    manifest = stage.stage_campaign(written(tmp_path, num_perf_trials=42))
    assert manifest["eval"] == {
        "num_correct_trials": 5, "num_perf_trials": 42, "timeout": 300,
    }


def test_the_manifest_records_the_config_so_a_zero_dedup_count_is_not_ambiguous(tmp_path):
    # "deduped: 0" reads two ways -- no duplicates, or dedup_by_code_sha1 off. §7 runs the off
    # arm as a control and §12 reports the hit rate off this number.
    manifest = stage.stage_campaign(written(tmp_path, dedup_by_code_sha1=False))
    assert manifest["deduped"] == 0 and manifest["config"]["dedup_by_code_sha1"] is False


def test_the_manifest_counts_the_evals_the_dedup_saved(tmp_path):
    # §12 reports the dedup hit rate off this; it is a saving, not a bug, and it is only
    # legible if the pass writes down what it collapsed.
    conf = written(tmp_path, per_problem={1: 3})
    rows = stage.read_rollouts(
        str(tmp_path / "campaign" / stage.ROLLOUTS / "unit_0.jsonl.gz")
    )
    stage.write_rollouts(
        [dataclasses.replace(r, code_sha1="dup") for r in rows],
        str(tmp_path / "campaign" / stage.ROLLOUTS / "unit_0.jsonl.gz"),
    )
    manifest = stage.stage_campaign(conf)
    assert manifest["rollouts"] == 3 and manifest["staged"] == 1 and manifest["deduped"] == 2


def test_the_manifest_counts_the_prefixes_no_part_ever_covered(tmp_path):
    # An array that lost tasks leaves the parts it did write, and this pass globs exactly those
    # -- the FileNotFoundError above fires only when there are none at all. Not a hard fail:
    # generate() legitimately drops a prefix with no budget left, so the count is the signal.
    conf = written(tmp_path, per_problem={1: 2})
    path = tmp_path / "campaign" / prefixes.PREFIXES
    path.write_text(
        path.read_text() + json.dumps(dataclasses.asdict(pre("p1s9", problem_id=1))) + "\n"
    )
    # A second rollout on a prefix that already has one, so the count cannot be read off the
    # rollouts: K of them share a prefix, and it is prefixes the shortfall is measured in.
    part = str(tmp_path / "campaign" / stage.ROLLOUTS / "unit_0.jsonl.gz")
    stage.write_rollouts(stage.read_rollouts(part) + [roll("p1s0__j01", sha="extra")], part)
    manifest = stage.stage_campaign(conf)
    assert manifest["rollouts"] == 3
    assert manifest["prefixes"] == 3 and manifest["prefixes_with_rollouts"] == 2


def test_a_unit_that_generated_nothing_plans_no_shards_rather_than_an_empty_one(tmp_path):
    # Reachable: a unit whose prefixes were all dropped for having no budget left writes a part
    # with no rows. An empty shard would be an eval job with a subset it staged nothing for.
    conf = written(tmp_path, parts=0)
    stage.write_rollouts([], stage.unit_path(conf.prm_rollout.out_dir, "empty_unit"))
    manifest = stage.stage_campaign(conf)
    assert manifest["shards"] == [] and manifest["staged"] == 0


def test_a_campaign_with_no_rollouts_yet_raises(tmp_path):
    conf = written(tmp_path, parts=0)
    with pytest.raises(FileNotFoundError, match=stage.ROLLOUTS):
        stage.stage_campaign(conf)


def test_the_gap_count_covers_problems_the_campaign_staged_nothing_for(tmp_path):
    # A campaign subsets problems, so a shard's range holds problems with no kernels at all --
    # and the harness still walks num_samples_per_problem slots for each of them. Counting only
    # the problems that were staged would report a shard as full when most of it is gaps.
    manifest = stage.stage_campaign(written(tmp_path, per_problem={1: 2, 5: 2}, eval_shards=1))
    shard = manifest["shards"][0]
    assert shard["subset"] == [1, 5] and shard["problems"] == 2
    assert shard["n_work"] == 10 and shard["n_staged"] == 4


def test_the_pass_reports_the_gaps_the_eval_jobs_will_walk(tmp_path):
    # The harness walks range(num_samples_per_problem) for every problem in the subset range,
    # so a shard whose problems are uneven pays for slots no kernel was staged under. It is
    # tolerated (a missing file is recorded as a compile failure, not a crash) but it is real
    # GPU time, and §12 cannot judge the shard plan without the number.
    conf = written(tmp_path, per_problem={1: 4, 2: 1}, eval_shards=1)
    manifest = stage.stage_campaign(conf)
    assert manifest["shards"][0]["n_work"] == 8 and manifest["staged"] == 5


def test_the_pass_picks_up_a_part_written_through_unit_path(tmp_path):
    # The seam between the two jobs: generation writes a unit's rows, staging globs them back.
    # Two spellings of the path would leave a unit generated and never staged, and the only
    # symptom would be a smaller campaign than the one that was paid for.
    conf = written(tmp_path, parts=0)
    out = conf.prm_rollout.out_dir
    stage.write_rollouts(assigned({1: 1}), stage.unit_path(out, "a_run__shard_00__round0"))
    assert stage.stage_campaign(conf)["staged"] == 1


def test_a_campaign_runs_from_a_config_file_on_the_command_line(tmp_path):
    conf = written(tmp_path, eval_shards=1)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({"prm_rollout": dataclasses.asdict(conf.prm_rollout)}))
    stage.main(["--config", str(path)])
    assert (tmp_path / "campaign" / stage.STAGE_MANIFEST).exists()


def test_a_campaign_spanning_two_levels_raises():
    # EvalConfig takes ONE level, and the map's (problem_id, sample_id) key is only unique
    # campaign-wide because of that. Two levels in one run dir would collide silently.
    rows = stage.assign(
        [roll("p1__j00"), roll("p2__j00")], where(("p1", 1, 1), ("p2", 2, 1))
    )
    with pytest.raises(ValueError, match="level"):
        stage.plan_shards(rows, cfg(eval_shards=1))
