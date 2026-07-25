"""``kernel_gen.core.cli``: the argparse groups, and the contract they must honour.

The load-bearing rule (cli.py's own docstring): **an argparse dest is a public YAML
key.** The config is persisted as ``dict(vars(args))`` and read back by a flat
``key: value`` scanner in ``triton_lint/runs.py`` that drops any line starting with a
space or a dash. So every flag must serialize to a single scalar -- ``nargs="+"`` is
forbidden, because PyYAML writes a list as a block whose lines all start with ``-``.
This file pins that contract, which no test covered before.
"""

from __future__ import annotations

import argparse

import pytest

from kernel_gen.core import cli


def _full_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    cli.add_dataset_args(parser)
    cli.add_model_args(parser)
    cli.add_sampling_args(parser)
    cli.add_prompt_args(parser)
    return parser


def _parse(argv: list[str]) -> argparse.Namespace:
    return _full_parser().parse_args(argv)


# -- the flat-scalar contract ----------------------------------------------


def test_every_flag_serializes_to_a_yaml_scalar_not_a_list():
    # The whole contract: dict(vars(args)) must be flat scalars, or runs.py's scanner
    # silently drops the key. A single list-valued flag would break it.
    args = _parse(["--model", "m", "--level", "1", "--all"])
    for key, value in vars(args).items():
        assert not isinstance(value, (list, tuple, dict, set)), f"{key} is not a scalar: {value!r}"


def test_comma_string_flags_stay_strings_not_nargs_lists():
    # --problems is the multi-value flag most tempting to make nargs="+"; it must stay a
    # comma string so it serializes on one line.
    args = _parse(["--model", "m", "--level", "1", "--problems", "1,5,10"])
    assert args.problems == "1,5,10"
    assert isinstance(args.problems, str)


def test_problems_and_rows_are_the_same_dest():
    # KernelBook is addressed by row; --rows is an alias, not a second key, or the
    # config would carry two names for one thing.
    assert _parse(["--model", "m", "--level", "5", "--rows", "0-9"]).problems == "0-9"
    assert _parse(["--model", "m", "--level", "1", "--problems", "3"]).problems == "3"


# -- defaults and dispatch -------------------------------------------------


def test_dataset_name_resolves_from_dataset_when_unset():
    args = _parse(["--model", "m", "--level", "1"])
    assert args.dataset == "kernelbench"
    assert args.dataset_name is None
    cli.resolve_dataset_name(args)
    assert args.dataset_name == cli.DATASET_DEFAULTS["kernelbench"]


def test_dataset_name_is_left_alone_when_given():
    args = _parse(["--model", "m", "--level", "5", "--dataset", "kernelbook",
                   "--dataset-name", "my/fork"])
    cli.resolve_dataset_name(args)
    assert args.dataset_name == "my/fork"


def test_sampling_defaults_match_the_documented_values():
    args = _parse(["--model", "m", "--level", "1"])
    assert args.num_samples == 10
    assert args.temperature == 0.3
    assert args.think_temperature == 1.0
    assert args.max_new_tokens == 16384
    assert args.max_model_len == 40960
    assert args.max_num_seqs == 32


def test_level_is_required():
    with pytest.raises(SystemExit):
        _full_parser().parse_args(["--model", "m"])
