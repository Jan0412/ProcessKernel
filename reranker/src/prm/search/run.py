"""CLI for the PRM-guided beam search: load the models, run it, write what it produced.

The artifact layout is the point. `pool/` is written under KernelBench's own naming so one
unmodified eval run grades every candidate the search finished -- not only the one the ORM
picked. That is what makes "could we have picked a better kernel" answerable at all, and
`report.py` is the join.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random

from kernel_gen.core.cli import DATASET_DEFAULTS
from kernel_gen.core.prompts import SYSTEM_PROMPT, build_base_prompt
from kernel_gen.core.sources import load_problems
from kernel_gen.core.text import ENTRY_CLASS, extract_code_block
from reranker.src.config import SEL_PRM, load_config
from reranker.src.prm.rollout import orm_score, rank_eval, rollout
from reranker.src.prm.rollout.encoding import PrefixEncoder
from reranker.src.prm.search import search as S

POOL, BEST, TREES = "pool", "best", "trees"
MANIFEST = "search_manifest.json"


def kernel_filename(level: int, problem_id: int, sample_id: int) -> str:
    """KernelBench's own name for a generated kernel -- eval_from_generations reads this."""
    return f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"


def _code(c) -> str:
    """The kernel inside a finished candidate, or '' when it never wrote one.

    `extract_code_block` still returns *something* for pure prose (its last-resort fallback
    is the dedented text itself, KGEN-9's `_largest_parseable_prefix` gate notwithstanding) --
    so a bare truthiness check on its return is not enough. Only a result that actually
    defines `ENTRY_CLASS` counts as a kernel; everything else is prose that never reached code.
    """
    try:
        code = extract_code_block(c.text)
    except Exception:  # noqa: BLE001 -- a malformed candidate must not lose the other 15
        return ""
    return code if ENTRY_CLASS in code else ""


def write_artifacts(result, orm_scores: dict, cfg, out_dir: str) -> dict:
    """Write pool/, best/ and trees/, and return the manifest (also written to disk)."""
    conf = cfg.prm_search
    for name in ((POOL, BEST, TREES) if conf.write_pool else (BEST, TREES)):
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    n_pool = n_no_code = 0
    for key, cands in sorted(result.pool.items()):
        level, pid = key
        graded = [(c, _code(c)) for c in cands]
        n_no_code += sum(1 for _, code in graded if not code)
        # Sample ids follow the pool's own order, NOT the ORM ranking: trees/ joins a pool
        # file back to its cid by sample_id, and that join must hold whatever the ORM says.
        graded = [(c, code) for c, code in graded if code]
        n_pool += len(graded)

        entries, codes = [], []
        for sample_id, (c, code) in enumerate(graded):
            if conf.write_pool:
                path = os.path.join(out_dir, POOL, kernel_filename(level, pid, sample_id))
                with open(path, "w") as fh:
                    fh.write(code)
            entries.append({
                "cid": c.cid, "sample_id": sample_id,
                "orm_score": orm_scores.get(c.cid), "stop": c.stop, "tokens": c.spent,
                "prm_scores": c.scores,
            })
            codes.append(code)

        best_i = max(
            range(len(entries)),
            key=lambda i: entries[i]["orm_score"] if entries[i]["orm_score"] is not None
            else float("-inf"),
        ) if entries else None
        picked = None
        if best_i is not None:
            picked = {"cid": entries[best_i]["cid"], "sample_id": entries[best_i]["sample_id"]}
            with open(os.path.join(out_dir, BEST, kernel_filename(level, pid, 0)), "w") as fh:
                fh.write(codes[best_i])

        steps = [
            {"step": step["step"],
             "candidates": [
                 {k: r[k] for k in ("cid", "parent", "score", "kept", "tokens",
                                    "cut_char", "reached_target", "done")}
                 for r in step["rows"] if r["pkey"] == key
             ]}
            for step in result.steps
        ]
        tree = {
            "level": level, "problem_id": pid,
            "name": cands[0].problem.name if cands else "",
            "steps": steps, "pool": entries, "picked": picked,
        }
        with open(os.path.join(out_dir, TREES, f"problem_{level}_{pid}.json"), "w") as fh:
            json.dump(tree, fh, indent=2)

    manifest = {
        "config": dataclasses.asdict(conf),
        "gen_model": cfg.prm_rollout.gen_model,
        "n_problems": len(result.pool),
        "n_pool": n_pool,
        "n_no_code": n_no_code,
        "n_pruned": sum(len(v) for v in result.pruned.values()),
        "n_steps": len(result.steps),
        "n_short_of_cut_target": sum(
            1 for step in result.steps for r in step["rows"] if not r["reached_target"]
        ),
    }
    with open(os.path.join(out_dir, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _prm(cfg):
    """`(scorer, encoder)` for the PRM, or `(None, None)` for the random control."""
    conf = cfg.prm_search
    if conf.selector != SEL_PRM:
        return None, None
    # load_scorer batches by cfg.train.per_device_eval_batch_size -- a trainer knob reached
    # from an inference path. Swap prm_batch_size in so the search owns its own batch size,
    # the same way _orm_scores swaps in the search's own ORM checkpoint.
    scoring_cfg = dataclasses.replace(
        cfg,
        train=dataclasses.replace(cfg.train, per_device_eval_batch_size=conf.prm_batch_size),
    )
    scorer, tokenizer = rank_eval.load_scorer(scoring_cfg, conf.prm_checkpoint)
    return scorer, PrefixEncoder(tokenizer, cfg.prm_rollout.max_length)


def _orm_scores(cfg, result) -> dict:
    """`cid -> ORM logit` over every finished, code-bearing candidate, in one batched pass.

    `orm_score.load_scorer` reads its `orm_*` fields off a `PRMRolloutConfig`-shaped object,
    not the whole `RerankerConfig` -- so `cfg.prm_rollout` is what gets passed, with only
    `orm_checkpoint` swapped for `prm_search`'s own (the search's pick-a-winner ORM is a
    separate concern from `prm_rollout.orm_checkpoint`, which imputes v3 training labels).
    Its encoding shape (`orm_max_length`, `orm_reserve_ref_tokens`) is not a `prm_search`
    knob, so it stays whatever the campaign's `prm_rollout` section carries.

    The reference comes off the Problem, never off disk: it is the exact text the model was
    prompted with. orm_score re-reads it from `data.kernelbench_dir` only because the
    label-imputation pipeline streams stored rows carrying no prompt -- that second path
    pointed at a tree with no level 1 or 2 and killed jobs 2474471-2 after a full hour.
    """
    conf = cfg.prm_search
    orm_conf = dataclasses.replace(cfg.prm_rollout, orm_checkpoint=conf.orm_checkpoint)
    scorer, encoder = orm_score.load_scorer(orm_conf)
    items, cids = [], []
    for _, cands in sorted(result.pool.items()):
        ref = cands[0].problem.ref_arch_src
        for c in cands:
            code = _code(c)
            if not code:
                continue
            items.append(encoder.encode(ref, code))
            cids.append(c.cid)
    if not items:
        return {}
    return dict(zip(cids, rank_eval.batched(items, scorer, conf.orm_batch_size)))


def main(argv=None) -> None:
    cfg = load_config(argv)
    conf = cfg.prm_search
    conf.validate()
    if conf.rounds != [0]:
        print(f"[WARN] prm_search.rounds={conf.rounds}: the PRM is trained on round 0 only, "
              "so its scores on later rounds are out of distribution")

    problems = load_problems(
        conf.dataset,
        ref_dir=conf.ref_dir,
        dataset_name=conf.dataset_name or DATASET_DEFAULTS[conf.dataset],
        level=conf.level,
        spec=conf.problems,
        all_rows=conf.problems is None,
    )
    backend = rollout._backend(cfg.prm_rollout, conf.gpu_memory_utilization)
    scorer, encoder = _prm(cfg)
    result = S.search(
        backend, problems, conf, cfg.prm_rollout, rollout.gen_counter(cfg.prm_rollout),
        scorer, encoder, SYSTEM_PROMPT, build_base_prompt, random.Random(conf.select_seed),
    )
    manifest = write_artifacts(result, _orm_scores(cfg, result), cfg, conf.out_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
