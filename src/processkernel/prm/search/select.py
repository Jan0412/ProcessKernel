"""PRM scores over live candidates, and the top-B cut that keeps the beam's parents diverse."""

from __future__ import annotations

from collections import Counter

from processkernel.config import SEL_RANDOM


def score(cands, conf, scorer, encoder, rng) -> list[float]:
    """One score per candidate, in order.

    `selector='random'` is the ablation control, and it must stay exactly this cheap: it is
    what separates "the PRM's ranking helped" from "branching and a wider ORM pool helped".
    """
    if conf.selector == SEL_RANDOM:
        return [rng.random() for _ in cands]
    if not cands:
        return []
    return scorer([encoder.encode_text(c.scored_prefix(conf)) for c in cands])


def top_b(cands, scores, conf):
    """`(kept, dropped)` -- the best `beam_width`, adjusted for the parent-diversity floor.

    With `beam_groups` > 1 each sub-beam keeps its own best `beam_width / beam_groups`, so a
    group is never displaced by a stronger one. One group is the plain beam.
    """
    order = sorted(range(len(cands)), key=lambda i: (-scores[i], cands[i].cid))
    width = conf.beam_width // conf.beam_groups
    floor = min(conf.min_distinct_parents, width)
    kept = []
    for g in sorted({cands[i].group for i in order}):
        mine = [i for i in order if cands[i].group == g]
        kept += _diversify(mine[:width], mine[width:], cands, floor)
    keep = set(kept)
    return [cands[i] for i in kept], [cands[i] for i in order if i not in keep]


def _diversify(kept, rest, cands, min_parents):
    """Swap the weakest kept out for the best unrepresented parent until the floor is met.

    The victim always has a sibling still in `kept`, so a swap can only raise the number of
    distinct parents, never lower it. When no such victim exists -- every kept candidate is
    the sole survivor of its parent -- the floor is already as high as this beam can go and
    the loop stops rather than trading one parent for another forever.
    """
    if min_parents <= 1:
        return kept
    kept = list(kept)
    for i in rest:
        parents = {cands[j].parent for j in kept}
        if len(parents) >= min_parents:
            break
        if cands[i].parent in parents:
            continue
        counts = Counter(cands[j].parent for j in kept)
        victim = next((j for j in reversed(kept) if counts[cands[j].parent] > 1), None)
        if victim is None:
            break
        kept = [j for j in kept if j != victim] + [i]
    return kept
