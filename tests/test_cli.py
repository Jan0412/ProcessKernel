"""``pk``: every command names a module that exists and runs as a program."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

from processkernel import cli


@pytest.mark.parametrize("command, module", sorted(cli.COMMANDS.items()))
def test_every_command_runs_a_module_with_a_main_guard(command, module):
    spec = importlib.util.find_spec(module)
    assert spec is not None, f"pk {command}: no module {module}"
    path = pathlib.Path(spec.origin)
    if path.name == "__init__.py":  # a package runs its __main__.py
        path = path.with_name("__main__.py")
    assert '__name__ == "__main__"' in path.read_text() or path.name == "__main__.py"


@pytest.mark.parametrize("command", [
    pytest.param(c, marks=pytest.mark.xfail(
        strict=True, reason="known bug: its pre-parser requires --config and --checkpoint"))
    if c == "rank-eval" else c
    for c in sorted(cli.COMMANDS)
])
# runpy re-runs a module other tests already imported; harmless for --help
@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_every_command_starts_and_prints_its_help(command, monkeypatch, capsys):
    # find_spec above never imports the module; this runs it, as `pk <command> --help` does.
    monkeypatch.setattr(sys, "argv", list(sys.argv))  # cli.main rebinds it
    with pytest.raises(SystemExit) as exit_:
        cli.main([command, "--help"])
    assert exit_.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_no_command_prints_the_list_and_fails(capsys):
    assert cli.main([]) == 2
    assert "train-prm" in capsys.readouterr().out


def test_help_succeeds(capsys):
    assert cli.main(["--help"]) == 0
    assert "search" in capsys.readouterr().out


def test_an_unknown_command_fails(capsys):
    assert cli.main(["grade"]) == 2
    assert "unknown command" in capsys.readouterr().err
