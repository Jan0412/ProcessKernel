"""The offline repair that rewrites a traced run to a deduplicated top-K (KGEN-25).

The fixture writes the *buggy* shape directly rather than driving the fake backend,
because the backend no longer produces it -- that is the point of the forward fix. What a
finished v4 run left on disk is a rectangular array holding K-1 distinct alternatives plus
a repeat of the sampled token, and this file pins what the repair must do to it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from kernel_gen.core.artifacts import append_jsonl, trace_dir
from kernel_gen.core.trace import PAD_ID, read_trace, write_trace

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
SCRIPT = os.path.join(REPO_ROOT, "scripts", "repair_traces.py")
VOCAB = 248320


def _buggy_trace(n_tokens: int = 6, k: int = 4, seed: int = 0):
    """A trace shaped the way the buggy pack() wrote them: K columns, one a repeat.

    The row holds the true top-(K-1) with the sampled token's copy re-inserted at its own
    rank, which is exactly what ``ordered[:k]`` produced from vLLM's K+1-wide input.
    """
    from kernel_gen.core.trace import TokenTrace

    rng = np.random.default_rng(seed)
    token_ids = rng.integers(10, 500, size=n_tokens).astype(np.int32)
    ids = np.full((n_tokens, k), PAD_ID, dtype=np.int32)
    lp = np.full((n_tokens, k), -np.inf, dtype=np.float16)
    sampled_lp = np.zeros(n_tokens, dtype=np.float32)
    sampled_rank = np.zeros(n_tokens, dtype=np.int16)

    for t in range(n_tokens):
        distinct = [int(token_ids[t])] + [int(x) for x in rng.integers(600, 999, size=k - 2)]
        rank = t % (k - 1)  # where the sampled token sits among the true alternatives
        row = sorted(distinct, key=lambda i: i != int(token_ids[t]))
        row = row[1:]
        row.insert(rank, int(token_ids[t]))
        values = [-0.1 - 0.6 * j for j in range(len(row))]
        # the duplicate: the sampled token's copy, at its own logprob
        row.insert(rank, int(token_ids[t]))
        values.insert(rank, values[rank])
        ids[t, : len(row)] = row[:k]
        lp[t, : len(row)] = values[:k]
        sampled_lp[t] = values[rank]
        sampled_rank[t] = rank + 1

    return TokenTrace(
        token_ids=token_ids, topk_ids=ids, topk_lp=lp, sampled_lp=sampled_lp,
        sampled_rank=sampled_rank, seg=np.zeros(n_tokens, dtype=np.int8),
        meta={"n_plan_tokens": 0, "n_code_tokens": n_tokens},
    )


@pytest.fixture
def run_dir(tmp_path):
    """A one-shard traced run carrying buggy traces, a journal, and both config files."""
    shard = tmp_path / "shard_00"
    target = trace_dir(str(shard), 0)
    os.makedirs(target)
    os.makedirs(shard / "rounds" / "round_0")

    records = []
    for sample in range(3):
        stem = f"level_6_problem_0_sample_{sample}_kernel"
        write_trace(os.path.join(target, f"{stem}.npz"), _buggy_trace(seed=sample))
        records.append({
            "stem": stem, "level": 6, "problem_id": 0, "sample_id": sample,
            "round": 0, "raw": f"raw {sample}", "code": f"code {sample}",
            "clean": True, "feedback": "", "findings": [],
            "trace": {"file": f"{stem}.npz", "n_plan_tokens": 0, "n_code_tokens": 6},
            "confidence": {"mean_margin": 0.0, "stale": True},
        })
    # An attempt whose trace failed to assemble: it must survive the rewrite untouched.
    records.append({
        "stem": "level_6_problem_0_sample_9_kernel", "level": 6, "problem_id": 0,
        "sample_id": 9, "round": 0, "raw": "", "code": "", "clean": False,
        "feedback": "", "findings": [], "trace": None, "confidence": {},
    })
    append_jsonl(os.path.join(target, "attempts.jsonl"), records)

    with open(os.path.join(shard, "traces", "trace_config.json"), "w") as fh:
        json.dump({"model": "m", "trace_topk": 4, "trace_window": 512,
                   "vocab_size": VOCAB, "logprobs_mode": "raw_logprobs"}, fh)
    for path in (shard / "generation_config.yaml",
                 shard / "rounds" / "round_0" / "generation_config.yaml"):
        with open(path, "w") as fh:
            yaml.safe_dump({"trace": True, "trace_topk": 4, "trace_window": 512,
                            "model": "m", "rounds": 3}, fh, sort_keys=True)
    return tmp_path


def _run(run_dir, *extra):
    result = subprocess.run(
        [sys.executable, SCRIPT, "--run-dir", str(run_dir), "--k", "3", "--force", *extra],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _journal(run_dir):
    path = os.path.join(trace_dir(str(run_dir / "shard_00"), 0), "attempts.jsonl")
    return [json.loads(line) for line in open(path)]


def test_the_fixture_really_reproduces_the_damage():
    # Without this, every assertion below would still pass against a duplicate-free
    # fixture -- the repair would simply truncate K to K-1 and the rows would come out
    # distinct anyway. This is what keeps the rest of the file from going vacuous.
    trace = _buggy_trace(seed=0)

    assert trace.topk_ids.shape[1] == 4
    assert all(len(set(row)) < len(row) for row in trace.topk_ids.tolist())
    assert all(
        row[int(rank) - 1] == int(token)
        for row, rank, token in zip(
            trace.topk_ids.tolist(), trace.sampled_rank, trace.token_ids
        )
    ), "the repeat must sit at the sampled token's own rank, as pack() left it"


def test_repair_leaves_k_distinct_alternatives_per_row(run_dir):
    _run(run_dir)

    target = trace_dir(str(run_dir / "shard_00"), 0)
    for name in sorted(os.listdir(target)):
        if not name.endswith(".npz"):
            continue
        trace = read_trace(os.path.join(target, name))
        assert trace.topk_ids.shape[1] == 3
        for row in trace.topk_ids.tolist():
            assert len(set(row)) == len(row), (name, row)
        assert trace.meta["trace_k"] == 3
        assert trace.meta["trace_repair"] == "kgen-25"


def test_repair_does_not_touch_what_was_already_correct(run_dir):
    target = trace_dir(str(run_dir / "shard_00"), 0)
    name = "level_6_problem_0_sample_0_kernel.npz"
    before = read_trace(os.path.join(target, name))

    _run(run_dir)

    after = read_trace(os.path.join(target, name))
    for field in ("token_ids", "sampled_lp", "sampled_rank", "seg"):
        assert np.array_equal(getattr(before, field), getattr(after, field)), field


def test_repair_recomputes_the_journal_confidence_in_place(run_dir):
    _run(run_dir)

    records = _journal(run_dir)
    assert len(records) == 4, "line count and order must be preserved"
    assert [r["stem"] for r in records] == sorted(r["stem"] for r in records)
    for record in records[:3]:
        assert "stale" not in record["confidence"], "the old block must be replaced"
        assert record["confidence"]["n_tokens"] == 6
        assert record["trace"]["trace_k"] == 3
        assert record["trace"]["file"] == f"{record['stem']}.npz", "the npz link must survive"
        assert record["raw"] == f"raw {record['sample_id']}", "unrelated fields untouched"


def test_an_attempt_without_a_trace_survives_the_rewrite(run_dir):
    _run(run_dir)

    untraced = _journal(run_dir)[-1]
    assert untraced["trace"] is None and untraced["confidence"] == {}


def test_repair_points_the_configs_at_the_k_actually_stored(run_dir):
    _run(run_dir)

    shard = run_dir / "shard_00"
    trace_config = json.load(open(os.path.join(shard, "traces", "trace_config.json")))
    assert trace_config["trace_topk"] == 3
    assert trace_config["trace_topk_requested"] == 4, "provenance: what was asked for"
    assert trace_config["vocab_size"] == VOCAB, "untouched keys survive"

    for path in (shard / "generation_config.yaml",
                 shard / "rounds" / "round_0" / "generation_config.yaml"):
        config = yaml.safe_load(open(path))
        assert config["trace_topk"] == 3 and config["trace_topk_requested"] == 4
        assert config["rounds"] == 3


def test_the_requested_k_recorded_is_the_one_the_run_asked_for(run_dir):
    # Not derivable from the array: the buggy pack() allocated exactly (n, k) and
    # truncated into it, so the stored width already IS the requested K -- the dropped
    # alternative left no gap. Deriving it as `width - 1` records the repaired K twice
    # and throws the provenance away, which is what the first pass over shard_00/03 did.
    _run(run_dir)

    target = trace_dir(str(run_dir / "shard_00"), 0)
    trace = read_trace(os.path.join(target, "level_6_problem_0_sample_0_kernel.npz"))
    assert trace.meta["trace_topk_requested"] == 4, "the run was launched with --trace-topk 4"
    assert trace.meta["trace_k"] == 3
    assert _journal(run_dir)[0]["trace"]["trace_topk_requested"] == 4


def test_a_wrong_provenance_stamp_is_corrected_without_touching_the_arrays(run_dir):
    # Recovery path for traces an earlier pass stamped wrongly: the arrays are already
    # right, so the file must be corrected in place rather than skipped as done.
    _run(run_dir)
    target = trace_dir(str(run_dir / "shard_00"), 0)
    path = os.path.join(target, "level_6_problem_0_sample_0_kernel.npz")
    good = read_trace(path)
    good.meta["trace_topk_requested"] = 3  # what the buggy first pass wrote
    write_trace(path, good)

    out = _run(run_dir)

    fixed = read_trace(path)
    assert "restamped" in out
    assert fixed.meta["trace_topk_requested"] == 4
    assert np.array_equal(fixed.topk_ids, good.topk_ids), "arrays must not be re-truncated"
    assert np.array_equal(fixed.topk_lp, good.topk_lp)
    assert _journal(run_dir)[0]["trace"]["trace_topk_requested"] == 4


def test_a_second_run_changes_nothing(run_dir):
    _run(run_dir)
    target = trace_dir(str(run_dir / "shard_00"), 0)
    fingerprint = {
        name: open(os.path.join(target, name), "rb").read()
        for name in sorted(os.listdir(target))
    }

    out = _run(run_dir)

    assert "already-repaired" in out
    assert fingerprint == {
        name: open(os.path.join(target, name), "rb").read()
        for name in sorted(os.listdir(target))
    }


def test_a_trace_written_by_the_fixed_pack_is_skipped_not_truncated(run_dir):
    # A K=4 trace carrying trace_k is native output, not damage. Truncating it to 3 would
    # discard a genuine alternative -- the exact loss the repair exists to bound.
    from kernel_gen.core.trace import pack

    target = trace_dir(str(run_dir / "shard_00"), 0)
    native = pack([5], [[(5, -0.2), (5, -0.2), (6, -1.0), (7, -2.0), (8, -3.0)]], k=4)
    write_trace(os.path.join(target, "level_6_problem_0_sample_0_kernel.npz"), native)

    out = _run(run_dir)

    after = read_trace(os.path.join(target, "level_6_problem_0_sample_0_kernel.npz"))
    assert "native-k4" in out
    assert after.topk_ids.shape[1] == 4 and after.meta["trace_k"] == 4


def test_dry_run_writes_nothing(run_dir):
    target = trace_dir(str(run_dir / "shard_00"), 0)
    before = {
        name: open(os.path.join(target, name), "rb").read()
        for name in sorted(os.listdir(target))
    }
    config_before = open(run_dir / "shard_00" / "generation_config.yaml", "rb").read()

    out = _run(run_dir, "--dry-run")

    assert "DRY RUN" in out and "repaired" in out
    assert before == {
        name: open(os.path.join(target, name), "rb").read()
        for name in sorted(os.listdir(target))
    }
    assert config_before == open(run_dir / "shard_00" / "generation_config.yaml", "rb").read()


def test_repair_leaves_no_temp_file_behind(run_dir):
    _run(run_dir)

    leftovers = [
        os.path.join(root, name)
        for root, _, names in os.walk(run_dir)
        for name in names
        if ".tmp" in name
    ]
    assert leftovers == []
