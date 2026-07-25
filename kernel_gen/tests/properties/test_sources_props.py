"""Property-based tests for ``parse_int_spec`` (Hypothesis).

The spec parser sits at the top of every run -- it turns ``--problems 1-49`` into the
indices that select dataset rows. An off-by-one or a dropped element here silently
generates the wrong problems, so the invariants (every emitted index lies in a requested
range; a range is inclusive; comma order is preserved) are worth pinning over all inputs.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from kernel_gen.core.text import parse_int_spec


@given(lo=st.integers(0, 500), length=st.integers(0, 200))
def test_a_range_is_inclusive_and_contiguous(lo, length):
    hi = lo + length
    assert parse_int_spec(f"{lo}-{hi}") == list(range(lo, hi + 1))


@given(values=st.lists(st.integers(0, 10000), min_size=1, max_size=30))
def test_a_comma_list_preserves_order_and_every_value(values):
    spec = ",".join(str(v) for v in values)
    assert parse_int_spec(spec) == values


@given(
    parts=st.lists(
        st.one_of(
            st.integers(0, 200).map(str),
            st.tuples(st.integers(0, 100), st.integers(0, 20)).map(lambda t: f"{t[0]}-{t[0] + t[1]}"),
        ),
        min_size=1,
        max_size=10,
    )
)
def test_every_emitted_index_belongs_to_some_requested_part(parts):
    # Whatever the mix of singletons and ranges, no index may appear that was not asked
    # for -- the failure mode that would generate a problem the user never selected.
    allowed = set()
    for part in parts:
        if "-" in part:
            a, b = part.split("-")
            allowed.update(range(int(a), int(b) + 1))
        else:
            allowed.add(int(part))

    result = parse_int_spec(",".join(parts))
    assert set(result) <= allowed
    assert all(isinstance(i, int) for i in result)


@given(spec=st.text(alphabet=st.characters(whitelist_categories=("Nd",), whitelist_characters=", "), max_size=20))
def test_only_raises_valueerror_on_malformed_comma_lists(spec):
    # Robustness on the shapes a human might type: digits, commas, spaces. Empty parts
    # (a trailing comma) are skipped; a part like "1 2" raises ValueError from int().
    # Nothing else should escape. (Ranges are excluded here on purpose -- an
    # astronomically large "0-10**19" is a MemoryError, a real but out-of-scope edge,
    # and the range shape is covered by the inclusive/contiguous test above.)
    try:
        parse_int_spec(spec)
    except ValueError:
        pass
