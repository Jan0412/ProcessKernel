"""Read a captured trace back and prove it supports the thing it was captured for.

Writing arrays to disk is easy; writing arrays that a process reward model can actually
be trained from is the claim, and this is what checks it. Given a run dir it rebuilds
one attempt's per-token frame, joins the linter's findings to it through their line
numbers, and prints the plan/code confidence split.

It is also the smoke test's read-out. Three things in the output are load-bearing and
worth reading every time:

``tail mass``
    Small but NONZERO. Zero would mean the top-20 truncation is not being measured at
    all -- the usual cause being a backend that returned no alternatives, which leaves
    every scalar quietly reading as maximum confidence.

``plan vs code mean confidence``
    Comparable between the halves. The plan is generated at temperature 1.0 and the code
    at 0.3-0.6, so if these are wildly apart the logprobs are ``processed_logprobs``
    after all and are measuring the CLI flags rather than the model.

``plans truncated``
    How many generations hit the token cap before reaching the code fence. Those kernels
    were written from a cut-off plan. It has always happened; it has never been counted.

Examples:
    uv run python -m kernel_gen.inspect_trace --run-dir runs/Qwen3.6-27B_level1_lintloop
    uv run python -m kernel_gen.inspect_trace --run-dir runs/... --stem level_1_problem_1_sample_0_kernel
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from kernel_gen.core import artifacts  # noqa: E402
from kernel_gen.core.trace import (  # noqa: E402
    SEG_CODE,
    SEG_PLAN,
    derive_scalars,
    rank1_calibration,
    read_trace,
)


def load_records(run_dir: str, round_index: int) -> list[dict]:
    return artifacts.read_jsonl(
        os.path.join(artifacts.trace_dir(run_dir, round_index), "attempts.jsonl")
    )


def check_calibration(run_dir: str, records: list[dict], round_index: int, limit: int = 20) -> None:
    """Prove from the data that the logprobs are pre-temperature. See rank1_calibration.

    Reads the arrays rather than the summaries, so it is the one part of the read-out
    that opens ``.npz`` files -- capped at ``limit`` traces, which is far more than
    enough for a rate over tens of thousands of tokens.
    """
    traced = [r for r in records if r["trace"]][:limit]
    if not traced:
        return

    columns: dict[int, list[np.ndarray]] = {SEG_PLAN: [], SEG_CODE: []}
    temperatures = {}
    for record in traced:
        path = os.path.join(artifacts.trace_dir(run_dir, round_index), record["trace"]["file"])
        trace = read_trace(path)
        temperatures[SEG_PLAN] = record["trace"].get("plan_temperature")
        temperatures[SEG_CODE] = record["trace"].get("code_temperature")
        for seg in (SEG_PLAN, SEG_CODE):
            mask = trace.seg == seg
            if mask.any():
                columns[seg].append((trace.topk_lp[mask], trace.sampled_rank[mask]))

    print(f"  calibration (over {len(traced)} traces):")
    for seg, name in ((SEG_PLAN, "plan"), (SEG_CODE, "code")):
        if not columns[seg]:
            continue
        lp = np.concatenate([c[0] for c in columns[seg]])
        rank = np.concatenate([c[1] for c in columns[seg]])
        recorded, observed = rank1_calibration(lp, rank)
        ratio = observed / recorded if recorded else float("nan")
        print(f"    {name} (T={temperatures[seg]}): recorded p1 {recorded:.4f}  "
              f"observed rank-1 {observed:.4f}  ratio {ratio:.3f}  (n={lp.shape[0]})")
    print("    ratio ~1.00 at T=1.0 and >1.00 at T<1.0 => the record is raw_logprobs")


def load_trace_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "traces", "trace_config.json")
    return json.loads(open(path).read()) if os.path.exists(path) else {}


def summarize_round(records: list[dict], round_index: int) -> None:
    """The health check over a whole round -- the numbers that say capture worked."""
    traced = [r for r in records if r["trace"]]
    print(f"\nround {round_index}: {len(records)} attempts, {len(traced)} traced")
    if not traced:
        return

    truncated = sum(1 for r in traced if r["trace"].get("plan_finish_reason") == "length")
    tokens = [r["confidence"].get("n_tokens", 0) for r in traced]
    least = [r["confidence"]["c_least"] for r in traced if "c_least" in r["confidence"]]

    print(f"  tokens/attempt   : {min(tokens)}-{max(tokens)}, mean {sum(tokens) / len(tokens):.0f}")
    print(f"  plans truncated  : {truncated}/{len(traced)} hit the cap before the fence")
    if least:
        print(f"  c_least          : {min(least):.2f}-{max(least):.2f}")
    clean = sum(1 for r in traced if r["clean"])
    print(f"  lint-clean       : {clean}/{len(traced)}")


def inspect(run_dir: str, record: dict, round_index: int, n_tokens: int) -> None:
    config = load_trace_config(run_dir)
    path = os.path.join(artifacts.trace_dir(run_dir, round_index), record["trace"]["file"])
    trace = read_trace(path)
    scalars = derive_scalars(
        trace.topk_lp, trace.sampled_lp, vocab_size=config.get("vocab_size")
    )
    meta = trace.meta

    print("=" * 78)
    print(f"{record['stem']}  (round {round_index}, {len(trace)} tokens, K={trace.k})")
    print(f"  model {config.get('model')}   logprobs_mode {config.get('logprobs_mode')}")
    print("=" * 78)

    # -- the seam ------------------------------------------------------------
    two_pass = meta.get("passes") == 2
    if two_pass:
        raw = record["raw"]
        plan_text = raw[meta["plan_char_start"] : meta["plan_char_end"]]
        print(f"\nseam: {meta['n_plan_tokens']} plan tokens + "
              f"{meta['n_code_tokens']} code tokens = {len(trace)}")
        print(f"      plan chars [{meta['plan_char_start']}:{meta['plan_char_end']}] "
              f"-> {plan_text[:60]!r}…")
        print(f"      plan finished on {meta.get('plan_finish_reason')!r} "
              f"(stop_reason {meta.get('plan_stop_reason')!r})")
    else:
        print(f"\nseam: single pass ({len(trace)} tokens), no plan/code split")

    # -- the check that raw_logprobs is really in effect ---------------------
    plan_mask, code_mask = trace.seg == SEG_PLAN, trace.seg == SEG_CODE
    if two_pass:
        print("\n                     plan (T=%s)   code (T=%s)"
              % (meta.get("plan_temperature"), meta.get("code_temperature")))
    else:
        print("\n                        (T=%s)" % meta.get("code_temperature"))
    for name in ("entropy", "deepconf_c", "tail_mass", "self_cert", "surprisal"):
        if name not in scalars:
            continue
        values = scalars[name]
        columns = f"{values[code_mask].mean():12.4f}"
        if two_pass:
            columns = f"{values[plan_mask].mean():12.4f} {columns}"
        print(f"  {name:<16s} {columns}")
    if two_pass:
        # The halves are NOT expected to match. Prose is genuinely less predictable than
        # Triton boilerplate, so a real gap here is the model, not a bug -- which is
        # exactly why the temperature question is settled by check_calibration's rank
        # statistic instead of by comparing these two columns.
        print("  (a gap here is prose vs code, not temperature -- see the calibration check)")

    # -- the join that credit assignment needs -------------------------------
    print(f"\nfindings ({len(record['findings'])}):")
    code = record["raw"][meta.get("code_char_start", 0) :]
    code_lines = code.splitlines()
    for finding in record["findings"]:
        lineno = finding.get("data", {}).get("lineno")
        source = code_lines[lineno - 1].strip() if lineno and lineno <= len(code_lines) else ""
        print(f"  {finding['severity']:<5s} {finding['check_id']:<6s} line {lineno} | {source[:50]}")
        print(f"        {finding['message'][:100]}")

    # -- the per-token frame -------------------------------------------------
    print(f"\nleast-confident {n_tokens} tokens (by deepconf_c):")
    order = scalars["deepconf_c"].argsort()[:n_tokens]
    print(f"  {'pos':>6s} {'seg':>5s} {'rank':>5s} {'entropy':>8s} {'deepconf':>9s} "
          f"{'tail':>8s} {'surprisal':>10s}")
    for position in sorted(order):
        seg = "plan" if trace.seg[position] == SEG_PLAN else "code"
        print(f"  {position:6d} {seg:>5s} {trace.sampled_rank[position]:5d} "
              f"{scalars['entropy'][position]:8.3f} {scalars['deepconf_c'][position]:9.3f} "
              f"{scalars['tail_mass'][position]:8.5f} "
              f"{scalars.get('surprisal', scalars['entropy'])[position]:10.3f}")

    print(f"\nsummary: {json.dumps(record['confidence'], indent=2)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--stem", default=None, help="default: the least confident attempt")
    parser.add_argument("--n-tokens", type=int, default=15)
    parser.add_argument("--all-rounds", action="store_true", help="summary only, every round")
    args = parser.parse_args()

    if args.all_rounds:
        for round_index in range(16):
            if not os.path.isdir(artifacts.trace_dir(args.run_dir, round_index)):
                break
            records = load_records(args.run_dir, round_index)
            summarize_round(records, round_index)
            check_calibration(args.run_dir, records, round_index)
        return

    records = load_records(args.run_dir, args.round)
    if not records:
        raise SystemExit(f"no traces in {artifacts.trace_dir(args.run_dir, args.round)}")
    summarize_round(records, args.round)
    check_calibration(args.run_dir, records, args.round)

    traced = [r for r in records if r["trace"]]
    if not traced:
        raise SystemExit("every attempt in this round is untraced")
    if args.stem:
        record = next((r for r in traced if r["stem"] == args.stem), None)
        if record is None:
            raise SystemExit(f"{args.stem} is not traced in round {args.round}")
    else:
        # The least confident trace is the interesting one, and picking it by default
        # is the whole argument for storing the summaries next to the arrays.
        record = min(traced, key=lambda r: r["confidence"].get("c_least", float("inf")))

    print()
    inspect(args.run_dir, record, args.round, args.n_tokens)


if __name__ == "__main__":
    main()
