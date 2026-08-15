"""``prm.rollout.dataset``: job D's lists -> the encoded lists a LambdaRank step reads.

Everything here runs against a character tokenizer and a campaign on tmp_path. No checkpoint,
no GPU, no download -- the same standard `test_rank_eval.py` holds itself to.
"""

from __future__ import annotations

import os

import pytest

from reranker.src.prm import build
from reranker.src.prm.rollout import dataset, lists
from reranker.tests.prm.rollout import campaignfixture
from reranker.tests.prm.rollout.campaignfixture import (
    RAW,
    RUN,
    SHARD,
    TAG,
    USER,
    CharTokenizer,
    campaign,
    pre,
    two_prefix_campaign,
)


def build_ds(cfg, split="val"):
    return dataset.PRMListDataset(cfg, split, CharTokenizer())


def test_a_list_becomes_one_item_carrying_its_prefixes_ids_and_rels(tmp_path):
    # The unit of iteration is the LIST, not the prefix: LambdaRank needs every item of a
    # group in the same batch, and `group_sizes` is what splits the scores back out.
    _, cfg = two_prefix_campaign(tmp_path)
    ds = build_ds(cfg)

    assert len(ds) == 1
    item = ds[0]
    assert item["rels"] == [2.0, 0.0]
    assert len(item["cand_input_ids"]) == 2


def test_each_prefix_is_encoded_as_the_stored_prompt_plus_its_own_cut(tmp_path):
    # v1 §2 decision 2: the PRM reads the prompt verbatim, then the generation so far. The two
    # prefixes of this list are cut at 4 and 12 characters of the same completion.
    _, cfg = two_prefix_campaign(tmp_path)
    tok = CharTokenizer()

    got = [tok.decode(ids) for ids in build_ds(cfg)[0]["cand_input_ids"]]
    assert got == [USER + RAW[:4], USER + RAW[:12]]


def test_the_lists_stay_in_file_order_so_the_eval_can_zip_scores_back_to_them(tmp_path):
    # _evaluate_lists runs a batch_size=1 unshuffled loader and pairs each batch with
    # self.lists[i] positionally. Reordering here would score every list against another
    # list's depth band, and every number would still compute.
    ps = [pre("p0", sid=0, cut_char=4), pre("p1", sid=1, cut_char=12)]
    rows = [
        lists.ListRow(list_key=f"{TAG}:2:37:0:{k}", run_tag=TAG, level=2, problem_id=37,
                      round=0, cut_index=k, rel_depth_mean=d, split="val", source="cut",
                      items=[lists.Item(f"p{i}", rel=1.0, n_rollouts=4, se=0.2)])
        for i, (k, d) in enumerate(((5, 0.2), (9, 0.8)))
    ]
    cfg = campaign(tmp_path, ps, rows)

    ds = build_ds(cfg)
    assert [lst.rel_depth_mean for lst in ds.lists] == [0.2, 0.8]


def test_a_list_citing_a_prefix_that_was_never_enumerated_raises_naming_it(tmp_path):
    # The text to score an item by lives on its prefix row. Absent, the list and the prefixes
    # come from different builds -- silently shortening the list would train on a subset.
    ps = [pre("p0", sid=0, cut_char=4)]
    row = lists.ListRow(
        list_key=f"{TAG}:2:37:0:5", run_tag=TAG, level=2, problem_id=37, round=0,
        cut_index=5, rel_depth_mean=0.5, split="val", source="cut",
        items=[lists.Item("p0", rel=2.0, n_rollouts=4, se=0.2),
               lists.Item("ghost", rel=0.0, n_rollouts=4, se=0.2)],
    )
    cfg = campaign(tmp_path, ps, [row])

    with pytest.raises(KeyError, match="ghost"):
        build_ds(cfg)


def test_a_prefix_whose_unit_has_no_v1_part_raises_naming_the_unit(tmp_path):
    # Same failure job E raises, through the same function: job A and the trainer would
    # otherwise be reading different builds.
    _, cfg = two_prefix_campaign(tmp_path)
    os.remove(os.path.join(str(tmp_path), build.PARTS, f"{RUN}__{SHARD}__round0.jsonl"))

    with pytest.raises(FileNotFoundError, match=f"{RUN}__{SHARD}__round0"):
        build_ds(cfg)


def test_the_encoder_is_given_the_prm_budget_and_never_the_orms(tmp_path):
    # ARCHITECTURE S4's trap. model.max_length bounds the ORM's ref + kernel; the PRM's budget
    # is prm_rollout.max_length, and it is the one job E encodes with -- so crossing them makes
    # the trainer's numbers and job E's incomparable while both still compute.
    _, cfg = two_prefix_campaign(tmp_path)
    cfg.prm_rollout.max_length = 5
    cfg.model.max_length = 4096

    assert build_ds(cfg).encoder.max_length == 5


def test_an_over_length_prefix_loses_its_head_and_still_ends_at_the_cut(tmp_path):
    # The whole reason PrefixEncoder exists: the tokenizer's default truncation drops the
    # TAIL, which is the cut point -- the single position the campaign measured.
    _, cfg = two_prefix_campaign(tmp_path)
    cfg.prm_rollout.max_length = 5
    tok = CharTokenizer()

    got = [tok.decode(ids) for ids in build_ds(cfg)[0]["cand_input_ids"]]
    assert got == [(USER + RAW[:4])[-5:], (USER + RAW[:12])[-5:]]


def test_the_train_split_is_read_when_it_is_asked_for(tmp_path):
    # One class, both splits: the trainer builds a train dataset and a val dataset from the
    # same campaign, and a split baked into the reader would need two.
    ps = [pre("p0", sid=0, cut_char=4, split="train")]
    row = lists.ListRow(
        list_key=f"{TAG}:2:37:0:5", run_tag=TAG, level=2, problem_id=37, round=0,
        cut_index=5, rel_depth_mean=0.5, split="train", source="cut",
        items=[lists.Item("p0", rel=1.5, n_rollouts=4, se=0.2)],
    )
    cfg = campaign(tmp_path, ps, [], train_rows=[row])

    assert build_ds(cfg, split="train")[0]["rels"] == [1.5]


def test_an_empty_split_is_refused_rather_than_training_on_nothing(tmp_path):
    # A zero-length dataset trains for zero steps and reports a finished run.
    ps = [pre("p0", sid=0, cut_char=4)]
    cfg = campaign(tmp_path, ps, [])

    with pytest.raises(ValueError, match="no lists"):
        build_ds(cfg)


def test_prefixes_cut_from_one_completion_share_a_single_loaded_source(tmp_path):
    # sources_for keys by completion, so N prefixes of one completion retain one copy of its
    # text rather than N. On the real corpus that is the difference between holding the
    # referenced completions and holding one per prefix.
    ps = [pre(f"p{i}", sid=0, cut_char=4 + i) for i in range(3)]
    row = lists.ListRow(
        list_key=f"{TAG}:2:37:0:5", run_tag=TAG, level=2, problem_id=37, round=0,
        cut_index=5, rel_depth_mean=0.5, split="val", source="cut",
        items=[lists.Item(f"p{i}", rel=float(i), n_rollouts=4, se=0.2) for i in range(3)],
    )
    # All three name sample 0, so they resolve to one completion and one part row.
    cfg = campaign(tmp_path, ps, [row], part_rows=[campaignfixture.part_row(0)])
    ds = build_ds(cfg)

    held = {id(src) for src in ds.sources.values()}
    assert len(ds.sources) == 3 and len(held) == 1


# --- the eval cap -------------------------------------------------------------------------


def many_lists(tmp_path, n, **over):
    """`n` val lists of two distinct prefixes each, cut at 4 and 12 characters."""
    ps, rows = [], []
    for k in range(n):
        a, b = f"p{2 * k}", f"p{2 * k + 1}"
        ps += [pre(a, sid=2 * k, cut_char=4), pre(b, sid=2 * k + 1, cut_char=12)]
        rows.append(lists.ListRow(
            list_key=f"{TAG}:2:37:0:{k}", run_tag=TAG, level=2, problem_id=37, round=0,
            cut_index=k, rel_depth_mean=0.5, split="val", source="cut",
            items=[lists.Item(a, rel=2.0, n_rollouts=4, se=0.2),
                   lists.Item(b, rel=0.0, n_rollouts=4, se=0.2)],
        ))
    return ps, campaign(tmp_path, ps, rows, **over)


def capped(cfg, **kw):
    return dataset.PRMListDataset(cfg, "val", CharTokenizer(), **kw)


def test_the_eval_cap_keeps_exactly_that_many_lists(tmp_path):
    _, cfg = many_lists(tmp_path, 10)
    assert len(capped(cfg, max_lists=4)) == 4


def test_no_cap_keeps_every_list(tmp_path):
    # 0 = every list, so adding the knob changed no existing run's number.
    _, cfg = many_lists(tmp_path, 5)
    assert len(capped(cfg, max_lists=0)) == 5


def test_a_cap_above_the_split_size_keeps_every_list(tmp_path):
    _, cfg = many_lists(tmp_path, 5)
    assert len(capped(cfg, max_lists=99)) == 5


def test_the_same_seed_draws_the_same_lists(tmp_path):
    # Two arms of a backbone comparison have to be scored on the SAME val subset, or their
    # eval_prm_ndcg values are not comparable and nothing in the output would say so.
    _, cfg = many_lists(tmp_path, 20)
    a, b = capped(cfg, max_lists=5), capped(cfg, max_lists=5)
    assert [x.list_key for x in a.lists] == [x.list_key for x in b.lists]


def test_a_different_seed_draws_a_different_subset(tmp_path):
    _, cfg = many_lists(tmp_path, 20)
    a = capped(cfg, max_lists=5, subsample_seed=1)
    b = capped(cfg, max_lists=5, subsample_seed=2)
    assert [x.list_key for x in a.lists] != [x.list_key for x in b.lists]


def test_the_subsample_stays_in_file_order(tmp_path):
    # `_evaluate_lists` zips self.lists against an unshuffled loader positionally, so the kept
    # lists must keep a stable order rather than the order random.sample happened to draw.
    _, cfg = many_lists(tmp_path, 20)
    idx = [x.cut_index for x in capped(cfg, max_lists=6).lists]
    assert idx == sorted(idx)


def test_the_cap_is_applied_before_the_v1_parts_are_read(tmp_path):
    # sources_for parses every part a referenced prefix touches and then holds a Source per
    # completion for the whole run. Subsampling afterwards pays both for lists it discards.
    _, cfg = many_lists(tmp_path, 10)
    ds = capped(cfg, max_lists=3)
    kept = {it.prefix_id for x in ds.lists for it in x.items}
    assert set(ds.sources) == kept
    assert set(ds.prefixes) == kept
