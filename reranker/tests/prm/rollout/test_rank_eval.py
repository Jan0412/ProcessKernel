"""``prm.rollout.rank_eval``: does an existing checkpoint rank the measured prefixes? (§6, §11)

Job E trains nothing -- one inference pass over the val lists, reporting pairwise accuracy and
NDCG **per depth bucket**. The depth slicing is the point: shallow prefixes have tiny true gaps
and maximal estimator noise, deep ones the reverse, and a single averaged number hides both.

Everything here runs against a stub scorer. No checkpoint, no GPU, no download.
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest
import torch
import yaml

from reranker.src.config import PRMRolloutConfig, RerankerConfig
from reranker.src.listwise.trainer import _graded_ndcg
from reranker.src.prm import build
from reranker.src.prm.rollout import encoding, lists, prefixes, rank_eval, rollout

TAG = "ar"
RUN, SHARD = "a_run", "shard_00"
SYS, USER = "SYSTEM PROMPT", "USER PROMPT\n"
RAW = "0123456789abcdefghij"


def cfg(**over) -> PRMRolloutConfig:
    base = PRMRolloutConfig(run_tags={"a_run": TAG}, baseline_timing_json=__file__)
    for k, v in over.items():
        setattr(base, k, v)
    base.validate()
    return base


def row(*rels, rel_depth_mean=0.5, key=None, cut_index=20) -> lists.ListRow:
    """One list, its items named ``<key>#i`` so a stub scorer can be keyed on them."""
    key = key or f"{TAG}:2:37:0:{cut_index}"
    return lists.ListRow(
        list_key=key,
        run_tag=TAG,
        level=2,
        problem_id=37,
        round=0,
        cut_index=cut_index,
        rel_depth_mean=rel_depth_mean,
        split="val",
        source="cut",
        items=[
            lists.Item(prefix_id=f"{key}#{i}", rel=r, n_rollouts=4, se=0.2)
            for i, r in enumerate(rels)
        ],
    )


def scores_of(row, *values) -> dict[str, float]:
    return {item.prefix_id: v for item, v in zip(row.items, values)}


# --- depth buckets -----------------------------------------------------------------------


def test_the_bands_split_the_configured_depth_window_evenly():
    assert rank_eval.bucket_edges(cfg(depth_buckets=4)) == [0.1, 0.3, 0.5, 0.7, 0.9]


def test_a_list_lands_in_the_band_its_mean_depth_falls_in():
    edges = rank_eval.bucket_edges(cfg(depth_buckets=4))
    assert [rank_eval.bucket_of(d, edges) for d in (0.1, 0.29, 0.3, 0.55, 0.89)] == [0, 0, 1, 2, 3]


def test_the_deepest_list_is_not_pushed_into_a_fifth_band_by_the_closed_top_edge():
    edges = rank_eval.bucket_edges(cfg(depth_buckets=4))
    assert rank_eval.bucket_of(0.9, edges) == 3


def test_a_depth_outside_the_window_is_clamped_into_a_band_rather_than_dropped():
    # It can only happen when the report's window has moved since the campaign was enumerated.
    # Every list must still be assigned -- but `report` counts these, so the mismatch is
    # visible rather than absorbed into the edge bands.
    edges = rank_eval.bucket_edges(cfg(depth_buckets=4))
    assert rank_eval.bucket_of(0.02, edges) == 0
    assert rank_eval.bucket_of(0.97, edges) == 3


# --- one list's numbers ------------------------------------------------------------------


def test_pairwise_accuracy_counts_the_ranking_pairs_the_scores_got_right():
    # rels 2, 1, 0 -> three pairs; scores 5, 9, 1 order the middle item above the top one,
    # so (0,1) is wrong and (0,2), (1,2) are right.
    got = rank_eval.list_metrics([5.0, 9.0, 1.0], [2.0, 1.0, 0.0])
    assert (got.pairs_correct, got.pairs_total) == (2, 3)


def test_a_perfect_order_scores_every_pair_and_a_reversed_one_scores_none():
    assert rank_eval.list_metrics([3.0, 2.0, 1.0], [2.0, 1.0, 0.0]).pairs_correct == 3
    assert rank_eval.list_metrics([1.0, 2.0, 3.0], [2.0, 1.0, 0.0]).pairs_correct == 0


def test_a_tie_in_the_scores_is_not_credited_as_a_correct_ranking():
    # `>`, matching listwise/trainer.py's own pairwise accuracy: a model that cannot separate
    # two prefixes has not ranked them.
    assert rank_eval.list_metrics([1.0, 1.0], [1.0, 0.0]).pairs_correct == 0


def test_a_list_with_no_ranking_pair_reports_no_pairs_rather_than_a_zero_accuracy():
    assert rank_eval.list_metrics([1.0, 2.0], [1.0, 1.0]).pairs_total == 0


def test_ndcg_is_the_statistic_the_listwise_trainer_already_computes():
    # Reused rather than restated (§6), so this plan's NDCG and a future trainer's are one
    # number -- not two implementations that agree until one of them is changed.
    scores, rels = [5.0, 9.0, 1.0], [2.0, 1.0, 0.0]
    assert rank_eval.list_metrics(scores, rels).ndcg == _graded_ndcg(
        torch.tensor(scores), torch.tensor(rels)
    )


# --- the report --------------------------------------------------------------------------


def test_every_band_is_reported_including_the_ones_no_list_landed_in():
    # An empty bucket reported as 0.0 reads as "the model ranks nothing right there"; it has
    # to be distinguishable from "nothing was measured there" (§6).
    conf = cfg(depth_buckets=4)
    shallow = row(2.0, 0.0, rel_depth_mean=0.15)
    got = rank_eval.report([shallow], scores_of(shallow, 9.0, 1.0), conf)

    assert [b["lo"] for b in got["buckets"]] == [0.1, 0.3, 0.5, 0.7]
    assert got["buckets"][0]["n_lists"] == 1
    for band in got["buckets"][1:]:
        assert band["n_lists"] == 0
        assert band["pairwise_acc"] is None and band["ndcg"] is None


def test_a_bands_accuracy_pools_its_pairs_rather_than_averaging_its_lists():
    # Pairs are the unit of evidence: a 4-item list carries 6 of them and a pair carries 1, so
    # a mean of per-list accuracies would weight the pair as heavily as the list.
    conf = cfg(depth_buckets=1)
    wide = row(2.0, 1.0, 0.0, rel_depth_mean=0.5)          # 3 pairs, all right
    narrow = row(1.0, 0.0, rel_depth_mean=0.5, key=f"{TAG}:2:38:0:20")   # 1 pair, wrong
    got = rank_eval.report(
        [wide, narrow],
        {**scores_of(wide, 3.0, 2.0, 1.0), **scores_of(narrow, 1.0, 2.0)},
        conf,
    )
    assert got["buckets"][0]["n_pairs"] == 4
    assert got["buckets"][0]["pairwise_acc"] == 0.75
    assert got["pairwise_acc"] == 0.75


def test_the_report_counts_every_list_it_was_given():
    conf = cfg(depth_buckets=4)
    rows = [row(2.0, 0.0, rel_depth_mean=d, key=f"{TAG}:2:{i}:0:20")
            for i, d in enumerate((0.15, 0.35, 0.55, 0.75, 0.85))]
    scores = {item.prefix_id: 1.0 for r in rows for item in r.items}
    got = rank_eval.report(rows, scores, conf)
    assert got["lists"] == 5
    assert sum(b["n_lists"] for b in got["buckets"]) == 5


def test_a_list_whose_depth_the_window_no_longer_covers_is_counted_not_hidden():
    conf = cfg(depth_buckets=4)
    stray = row(2.0, 0.0, rel_depth_mean=0.02)
    got = rank_eval.report([stray], scores_of(stray, 9.0, 1.0), conf)
    assert got["lists_outside_window"] == 1


def test_two_item_lists_are_reported_apart_because_a_corpus_of_pairs_is_a_pairwise_dataset():
    conf = cfg(depth_buckets=1)
    pair = row(1.0, 0.0, rel_depth_mean=0.5)
    wide = row(2.0, 1.0, 0.0, rel_depth_mean=0.5, key=f"{TAG}:2:38:0:20")
    got = rank_eval.report(
        [pair, wide],
        {**scores_of(pair, 2.0, 1.0), **scores_of(wide, 3.0, 2.0, 1.0)},
        conf,
    )
    assert got["two_item_lists"] == 1


def test_an_item_the_scorer_never_saw_raises_rather_than_ranking_the_rest_of_its_list():
    # Matched on the guard's own wording, not merely on KeyError: indexing `scores` would
    # raise anyway, with a bare id and no hint that a list was about to be scored short.
    conf = cfg(depth_buckets=1)
    r = row(2.0, 0.0, rel_depth_mean=0.5)
    with pytest.raises(KeyError, match="never scored"):
        rank_eval.report([r], {r.items[0].prefix_id: 1.0}, conf)


# --- batching: the scores must come back on the items they belong to ---------------------


class Forward:
    """A fake forward pass that scores by content, and records the batches it was handed."""

    def __init__(self):
        self.seen: list[list[int]] = []

    def __call__(self, batch: list[list[int]]) -> list[float]:
        self.seen.append([len(ids) for ids in batch])
        return [float(len(ids)) for ids in batch]


def test_scores_come_back_on_the_sequences_they_were_computed_for():
    # Length-sorting is what keeps padding off a 16k budget, and it is also the classic way
    # to hand a list back permuted. The scores must be in the caller's order, not the batch's.
    forward = Forward()
    got = rank_eval.batched([[0], [0] * 3, [0] * 2, [0] * 4], forward, batch_size=2)
    assert got == [1.0, 3.0, 2.0, 4.0]


def test_a_batch_holds_sequences_of_similar_length_and_never_more_than_batch_size():
    forward = Forward()
    rank_eval.batched([[0], [0] * 3, [0] * 2, [0] * 4], forward, batch_size=2)
    assert forward.seen == [[4, 3], [2, 1]]


def test_nothing_is_scored_when_there_is_nothing_to_score():
    forward = Forward()
    assert rank_eval.batched([], forward, batch_size=2) == []
    assert forward.seen == []


# --- the pass ----------------------------------------------------------------------------


class CharTokenizer:
    """One id per character, so a test can read back exactly what the scorer was shown."""

    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


def pre(pid: str, *, sid: int, cut_char: int, split="val", **over) -> prefixes.Prefix:
    fields = dict(
        prefix_id=pid, source="cut", run_name=RUN, run_tag=TAG, shard=SHARD, round=0,
        level=2, problem_id=37, sample_id=sid,
        stem=f"level_2_problem_37_sample_{sid}_kernel",
        cut_char=cut_char, cut_index=5, cut_kind="code", n_cuts_total=20, rel_depth=0.5,
        list_key=f"{TAG}:2:37:0:5", split=split, selection="random", selection_score=None,
        K=4, min_rollouts=2,
    )
    fields.update(over)
    return prefixes.Prefix(**fields)


def part_row(sid: int, **over) -> dict:
    r = {
        "run_name": RUN, "shard": SHARD, "round": 0, "level": 2, "problem_id": 37,
        "sample_id": sid, "stem": f"level_2_problem_37_sample_{sid}_kernel",
        build.ROW_SHA1: "a" * 40, "prompt": USER, "raw": RAW,
        "cuts": [4, 10], "cut_kinds": ["prose", "code"], "cut_index": [0, 1], "correct": True,
    }
    r.update(over)
    return r


def campaign(tmp_path, ps, list_rows, *, part_rows=None, train_rows=()) -> RerankerConfig:
    """v1's parts plus job A's and job D's outputs, laid out as the campaign writes them."""
    parts = tmp_path / build.PARTS
    parts.mkdir(exist_ok=True)
    (parts / f"{RUN}__{SHARD}__round0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in (part_rows or [part_row(p.sample_id) for p in ps]))
    )
    (tmp_path / build.PROMPTS).write_text(json.dumps({"a" * 40: SYS}))

    out = tmp_path / "campaign"
    out.mkdir(exist_ok=True)
    (out / prefixes.PREFIXES).write_text(
        "".join(json.dumps(dataclasses.asdict(p)) + "\n" for p in ps)
    )
    for split, rows in (("val", list_rows), ("train", train_rows)):
        (out / lists.LISTS.format(split=split)).write_text(
            "".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in rows)
        )
    return RerankerConfig(prm_rollout=cfg(
        parts_glob=os.path.join(str(tmp_path), build.PARTS, "*.jsonl"),
        out_dir=str(out),
        depth_buckets=1,
    ))


def two_prefix_campaign(tmp_path, rel_depth_mean=0.5, **over):
    """One val list of two prefixes cut at 4 and 12 characters."""
    ps = [pre("p0", sid=0, cut_char=4), pre("p1", sid=1, cut_char=12)]
    lst = lists.ListRow(
        list_key=f"{TAG}:2:37:0:5", run_tag=TAG, level=2, problem_id=37, round=0,
        cut_index=5, rel_depth_mean=rel_depth_mean, split="val", source="cut",
        items=[lists.Item("p0", rel=2.0, n_rollouts=4, se=0.2),
               lists.Item("p1", rel=0.0, n_rollouts=4, se=0.2)],
    )
    return ps, campaign(tmp_path, ps, [lst], **over)


class StubScorer:
    """Scores by the length of what it was shown, and keeps every sequence for inspection."""

    def __init__(self):
        self.seen: list[list[int]] = []

    def __call__(self, ids: list[list[int]]) -> list[float]:
        self.seen.extend(ids)
        return [float(len(x)) for x in ids]


def test_the_pass_scores_the_val_lists_and_reports_them(tmp_path):
    _, conf = two_prefix_campaign(tmp_path)
    got = rank_eval.rank_eval(conf, scorer=StubScorer(), tokenizer=CharTokenizer())

    # p0's text is shorter than p1's, so the stub ranks p1 above p0 -- and p0 is the relevant
    # one, so the single ranking pair is wrong.
    assert got["lists"] == 1 and got["items"] == 2 and got["pairs"] == 1
    assert got["pairwise_acc"] == 0.0


def test_the_scorer_is_shown_the_stored_prompt_and_the_prefix_and_nothing_else(tmp_path):
    ps, conf = two_prefix_campaign(tmp_path)
    scorer = StubScorer()
    rank_eval.rank_eval(conf, scorer=scorer, tokenizer=CharTokenizer())

    src = rollout.Source(system_prompt=SYS, prompt=USER, raw=RAW)
    assert [CharTokenizer().decode(ids) for ids in scorer.seen] == [
        encoding.scored_text(p, src) for p in ps
    ]


def test_the_train_lists_are_not_scored_because_only_val_is_unbiased(tmp_path):
    # N3 fixes val selection to random; the train half may be scored-selected in a later
    # campaign, and a number computed over it would be inflated with no way to tell.
    ps = [pre("p0", sid=0, cut_char=4), pre("t0", sid=1, cut_char=12, split="train")]
    val = lists.ListRow(
        list_key=f"{TAG}:2:37:0:5", run_tag=TAG, level=2, problem_id=37, round=0,
        cut_index=5, rel_depth_mean=0.5, split="val", source="cut",
        items=[lists.Item("p0", rel=2.0, n_rollouts=4, se=0.2)],
    )
    train = dataclasses.replace(
        val, split="train", items=[lists.Item("t0", rel=1.0, n_rollouts=4, se=0.2)]
    )
    conf = campaign(tmp_path, ps, [val], train_rows=[train])
    scorer = StubScorer()
    got = rank_eval.rank_eval(conf, scorer=scorer, tokenizer=CharTokenizer())

    assert got["lists"] == 1 and got["items"] == 1
    assert len(scorer.seen) == 1


def test_a_list_item_with_no_prefix_row_raises_rather_than_shortening_its_list(tmp_path):
    ps = [pre("p0", sid=0, cut_char=4)]
    lst = lists.ListRow(
        list_key=f"{TAG}:2:37:0:5", run_tag=TAG, level=2, problem_id=37, round=0,
        cut_index=5, rel_depth_mean=0.5, split="val", source="cut",
        items=[lists.Item("p0", rel=2.0, n_rollouts=4, se=0.2),
               lists.Item("gone", rel=0.0, n_rollouts=4, se=0.2)],
    )
    conf = campaign(tmp_path, ps, [lst])
    with pytest.raises(KeyError, match="gone"):
        rank_eval.rank_eval(conf, scorer=StubScorer(), tokenizer=CharTokenizer())


def test_the_report_lands_beside_the_lists_it_scored(tmp_path):
    _, conf = two_prefix_campaign(tmp_path)
    got = rank_eval.rank_eval(
        conf, scorer=StubScorer(), tokenizer=CharTokenizer(), checkpoint="/ckpt/v1"
    )
    written = json.loads(
        (tmp_path / "campaign" / rank_eval.REPORT).read_text()
    )
    assert written["pairwise_acc"] == got["pairwise_acc"]
    assert written["checkpoint"] == "/ckpt/v1"
    assert written["buckets"] == got["buckets"]


def test_a_prefix_whose_v1_row_is_gone_names_the_build_it_could_not_be_found_in(tmp_path):
    ps = [pre("p0", sid=0, cut_char=4)]
    lst = lists.ListRow(
        list_key=f"{TAG}:2:37:0:5", run_tag=TAG, level=2, problem_id=37, round=0,
        cut_index=5, rel_depth_mean=0.5, split="val", source="cut",
        items=[lists.Item("p0", rel=2.0, n_rollouts=4, se=0.2)],
    )
    conf = campaign(tmp_path, ps, [lst], part_rows=[part_row(9)])
    with pytest.raises(KeyError, match="p0"):
        rank_eval.rank_eval(conf, scorer=StubScorer(), tokenizer=CharTokenizer())


def test_a_batch_size_below_one_is_refused_rather_than_raising_out_of_range():
    with pytest.raises(ValueError, match="batch"):
        rank_eval.batched([[0], [0]], Forward(), batch_size=0)


def test_a_scorer_that_returns_too_few_scores_is_refused_rather_than_zipped_short(tmp_path):
    # zip stops at the shorter side, so the tail of the corpus would go unscored and the
    # report would be computed over a subset while naming the whole one.
    _, conf = two_prefix_campaign(tmp_path)
    with pytest.raises(ValueError, match="scores"):
        rank_eval.rank_eval(conf, scorer=lambda ids: [1.0], tokenizer=CharTokenizer())


def test_a_scorer_arrives_with_the_tokenizer_that_encoded_for_it_or_not_at_all(tmp_path):
    _, conf = two_prefix_campaign(tmp_path)
    with pytest.raises(ValueError, match="tokenizer"):
        rank_eval.rank_eval(conf, scorer=StubScorer())


def test_a_scored_val_prefix_cannot_be_read_back_so_the_number_cannot_be_inflated(tmp_path):
    # §12's stop-gate: "any val prefix is not random -> the v1-PRM ranking number is inflated
    # and must not be reported". N3 lives in `Prefix.__post_init__`, so reading the file is
    # already the check -- job E cannot score such a campaign even by accident.
    ps, conf = two_prefix_campaign(tmp_path)
    rows = [dataclasses.asdict(p) for p in ps]
    rows[0] |= {"selection": "entropy", "selection_score": 0.83}
    (tmp_path / "campaign" / prefixes.PREFIXES).write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    with pytest.raises(ValueError, match="N3"):
        rank_eval.rank_eval(conf, scorer=StubScorer(), tokenizer=CharTokenizer())


def test_a_prefix_whose_unit_has_no_v1_part_names_the_unit_rather_than_scoring_without_it(tmp_path):
    ps, conf = two_prefix_campaign(tmp_path)
    os.remove(os.path.join(str(tmp_path), build.PARTS, f"{RUN}__{SHARD}__round0.jsonl"))
    with pytest.raises(FileNotFoundError, match=f"{RUN}__{SHARD}__round0"):
        rank_eval.sources_for(conf.prm_rollout, ps)


# --- the checkpoint half: what can be checked without one --------------------------------


def test_scoring_needs_a_checkpoint_because_there_is_no_default_worth_guessing():
    with pytest.raises(ValueError, match="checkpoint"):
        rank_eval.load_scorer(RerankerConfig(), None)


def test_the_head_is_read_off_the_checkpoint_that_saved_it(tmp_path):
    # Checkpoint-agnostic means exactly this: how a backbone's outputs become one scalar is
    # read from what the training run wrote down, not guessed from the config in hand.
    (tmp_path / "reranker_head.json").write_text(
        json.dumps({"head_type": "yes_no_lm", "yes_id": 7, "no_id": 9})
    )
    head = rank_eval.head_of(RerankerConfig(), str(tmp_path), CharTokenizer())
    assert (head.head_type, head.yes_id, head.no_id) == ("yes_no_lm", 7, 9)


def test_a_checkpoint_that_saved_no_head_falls_back_to_the_configured_one(tmp_path):
    cfg_ = RerankerConfig()
    cfg_.model.head_type = "seq_cls"
    assert rank_eval.head_of(cfg_, str(tmp_path), CharTokenizer()).head_type == "seq_cls"


def test_a_configured_yes_no_head_resolves_its_two_vocabulary_ids(tmp_path):
    class OneTokenTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [{"yes": 5, "no": 6}.get(text.strip().lower(), 0)]

    cfg_ = RerankerConfig()
    cfg_.model.head_type = "yes_no_lm"
    head = rank_eval.head_of(cfg_, str(tmp_path), OneTokenTokenizer())
    assert (head.yes_id, head.no_id) == (5, 6)


def test_the_cli_scores_the_named_checkpoint_and_writes_the_report(tmp_path, monkeypatch, capsys):
    # The one seam a suite without a GPU cannot cross: `load_scorer` is what needs the model.
    # Everything on either side of it -- argument parsing, the pass, the printed table -- is
    # the real code path.
    _, conf = two_prefix_campaign(tmp_path)
    monkeypatch.setattr(
        rank_eval, "load_scorer", lambda cfg, ckpt: (StubScorer(), CharTokenizer())
    )
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({"prm_rollout": dataclasses.asdict(conf.prm_rollout)}))
    rank_eval.main(["--config", str(path), "--checkpoint", "/ckpt/v1"])

    printed = capsys.readouterr().out
    assert "1 val lists" in printed
    written = json.loads((tmp_path / "campaign" / rank_eval.REPORT).read_text())
    assert written["checkpoint"] == "/ckpt/v1"


def test_an_empty_band_prints_a_dash_rather_than_a_number_it_did_not_measure(tmp_path, monkeypatch, capsys):
    _, conf = two_prefix_campaign(tmp_path, )
    conf.prm_rollout.depth_buckets = 4          # the list sits at 0.5; three bands stay empty
    monkeypatch.setattr(
        rank_eval, "load_scorer", lambda cfg, ckpt: (StubScorer(), CharTokenizer())
    )
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({"prm_rollout": dataclasses.asdict(conf.prm_rollout)}))
    rank_eval.main(["--config", str(path), "--checkpoint", "/ckpt/v1"])
    assert "--" in capsys.readouterr().out


def test_a_campaign_cut_under_a_different_depth_window_says_so_out_loud(tmp_path, monkeypatch, capsys):
    # The clamp keeps every list assigned; the warning is what stops the bands being read as
    # if they were the ones the campaign was enumerated under.
    _, conf = two_prefix_campaign(tmp_path, rel_depth_mean=0.02)
    monkeypatch.setattr(
        rank_eval, "load_scorer", lambda cfg, ckpt: (StubScorer(), CharTokenizer())
    )
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({"prm_rollout": dataclasses.asdict(conf.prm_rollout)}))
    rank_eval.main(["--config", str(path), "--checkpoint", "/ckpt/v1"])
    assert "WARNING" in capsys.readouterr().out
