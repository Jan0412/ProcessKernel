"""Rewrite a generated kernel's launch constants to a given config.

We edit the raw source text at spans located by the AST, applied back-to-front so earlier
offsets stay valid. Two alternatives were rejected:

  * regex -- ``BLOCK_SIZE`` matches inside ``BLOCK_SIZE_M``, and grid expressions mention
    the same names, so a textual substitution cannot tell a definition from a use.
  * ``ast.unparse`` -- rewrites the whole file, drops comments, and reformats. The identity
    config would then not round-trip, which we rely on: the sweep re-measures the kernel's
    own constants as config 0, and that measurement is only a fair denominator for the
    tuning gain if the source is byte-identical to the one already evaluated.
"""

from __future__ import annotations

import ast

from autotune.knobs import LAUNCH_KNOBS, Site, TunabilityReport, analyze


class Unpatchable(RuntimeError):
    """The config could not be applied to this source; the sweep skips and records it."""


def patch_source(src: str, config: dict[str, int], report: TunabilityReport | None = None) -> str:
    """Apply ``config`` to ``src``. An empty config returns the source unchanged."""
    if not config:
        return src  # identity: config 0 must be byte-identical, see module docstring
    rep = report if report is not None else analyze(src)
    if rep.parse_error:
        raise Unpatchable(f"source does not parse: {rep.parse_error}")

    edits: list[tuple[int, int, str]] = []
    to_insert: dict[str, int] = {}  # launch knobs the source never mentions

    for name, value in config.items():
        if name in LAUNCH_KNOBS:
            sites = rep.launch_knob_sites.get(name)
            if sites:  # already on the launch: replace its value in place
                edits += [(s.start, s.end, str(value)) for s in sites]
            else:
                to_insert[name] = value
            continue

        knob = rep.knob(name)
        if knob is None:
            raise Unpatchable(f"{name!r} is not a tunable knob in this source")
        edits += [(s.start, s.end, str(value)) for s in knob.sites]

    # num_warps and num_stages both want the same insertion point -- just inside the launch's
    # closing paren. Emitting them as two zero-width edits at one offset splices them into
    # each other ("num_stages=2num_warps=8"), so they go in as a single edit.
    if to_insert:
        added = ", ".join(f"{name}={value}" for name, value in sorted(to_insert.items()))
        for site in rep.launch_insert_sites:
            # site.what carries the tokenizer's verdict on whether the argument list already
            # ends in a comma. See knobs._insert_point -- neither the raw text nor the AST
            # can answer that reliably.
            text = f", {added}" if site.what == "insert," else added
            edits.append((site.start, site.end, text))

    # Two knobs can be backed by one variable:
    #
    #     BLOCK = 64
    #     kernel[grid](..., BLOCK=BLOCK, BLOCK_SIZE=BLOCK)
    #
    # Both then resolve to the same assignment span, and writing it twice would splice the
    # second value into the first. Collapse duplicates; a genuine disagreement about what one
    # site should hold is unpatchable, not something to silently pick a winner for.
    merged: dict[tuple[int, int], str] = {}
    for start, end, text in edits:
        if merged.setdefault((start, end), text) != text:
            raise Unpatchable(
                f"two knobs share one site [{start}:{end}] but want different values "
                f"({merged[(start, end)]!r} vs {text!r})"
            )

    # Back-to-front so each edit's offsets are still valid when it is applied.
    out = src
    for (start, end), text in sorted(merged.items(), key=lambda e: e[0][0], reverse=True):
        out = out[:start] + text + out[end:]

    _verify(out, config)
    return out


def _verify(patched: str, config: dict[str, int]) -> None:
    """The patched source must parse, and must actually carry the values we asked for.

    This is what turns a subtle mis-patch (a value written to the wrong site, a knob whose
    grid expression silently disagrees) into a loud failure instead of a plausible-looking
    but meaningless timing.
    """
    try:
        ast.parse(patched)
    except SyntaxError as e:
        raise Unpatchable(f"patched source does not parse: {e}") from e

    rep = analyze(patched)
    for name, value in config.items():
        if name in LAUNCH_KNOBS:
            got = rep.launch_knob_values.get(name)
            if got != value:
                raise Unpatchable(f"{name}: wanted {value}, patched source reads {got!r}")
            continue
        knob = rep.knob(name)
        if knob is None:
            raise Unpatchable(f"{name!r} vanished from the patched source")
        if knob.current != value:
            raise Unpatchable(f"{name}: wanted {value}, patched source reads {knob.current}")
