"""The lint check registry.

Every check lives in its own file under ``family1/`` or ``family2/`` and registers itself
with :meth:`Registry.add`. Adding a family is a new subpackage plus one import at the
bottom of this file.

``CHECKS`` and ``run_checks`` are aliases onto the registry: they are what
``checker/__init__.py`` imports, so rebinding them here is what lets the collection stop
being a module-global list without the public API noticing.
"""

from __future__ import annotations

from ...core.check import Registry

LINT_REGISTRY = Registry("lint")

CHECKS = LINT_REGISTRY.checks
run_checks = LINT_REGISTRY.run


from .family1 import *  # noqa: E402,F401,F403  (populates LINT_REGISTRY)
from .family2 import *  # noqa: E402,F401,F403
