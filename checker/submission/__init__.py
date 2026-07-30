"""Can the evaluator load this file and call it?

The linter answers "is this good Triton?". That is not the same question as "can Python
import this module and instantiate ``ModelNew``", and the gap is not hypothetical: 3% of
the kernels the lint loop declared clean cannot be loaded at all, so they were labelled
good, stopped their repair loop, and scored zero.

This package answers only the loadability question. A kernel that compiles, imports,
instantiates and runs but computes entirely the wrong thing passes every check here, and
that is deliberate -- conflating "loadable" with "correct" would make the gate
unfalsifiable, and correctness is what eval is for.

**Why ``compile()`` and not the Triton compiler.** Three reasons, each verified rather
than assumed:

1. Compiling a Triton kernel needs a concrete signature -- argument dtypes and the
   *values* of every ``constexpr`` -- which come from the call site in ``forward`` and
   depend on the real tensors. Getting them means running ``forward``, which is eval.
2. ``@triton.jit`` cannot be built from ``exec``'d source at all: ``JITFunction.__init__``
   calls ``inspect.getsourcelines`` and raises *"@jit functions should be defined in a
   Python file"*.
3. There is no usable Triton driver on a login node, so it could never run in CI; and on a
   GPU node vLLM owns the card, so compiling generated Triton in-process risks killing a
   job holding hours of work.

``compile()`` is stdlib, needs no GPU, costs microseconds, and is exactly the necessary
condition for ``exec(compile(...))`` -- which is what the evaluator does. It catches every
failure measured in the corpus.
"""

from __future__ import annotations

from ..core.analyzer import Analyzer
from ..core.model import ModuleModel
from .checks import SUBMISSION_REGISTRY


class SubmissionAnalyzer(Analyzer):
    registry = SUBMISSION_REGISTRY

    def build(self, source: str, path: str = "") -> ModuleModel:
        """The shared skeleton, plus the question ``ast.parse`` does not ask.

        ``compile`` runs the symbol-table and codegen passes on top of the parser, and
        rejects a whole class of source the AST accepts -- a duplicated parameter name, a
        ``return`` outside a function, a starred assignment target. Those files parse, lint
        clean, and then fail at import.
        """
        model = super().build(source, path)

        # Asked unconditionally, including when ast.parse already failed: CPython's own
        # message is better feedback than the front end's note, and an empty file compiles
        # fine, so "no tree" is not by itself a compile failure. That is S1.1's business.
        try:
            compile(source, path or "<generated>", "exec")
        except Exception as exc:  # noqa: BLE001 - see S1.0: the family is wider than SyntaxError
            model.compile_error = str(exc)
            model.compile_error_lineno = getattr(exc, "lineno", None)

        return model

    def should_run_checks(self, model: ModuleModel) -> bool:
        """Always. A file that does not parse is what this gate exists to report, so
        unlike the linter it has plenty to say about one."""
        return True
