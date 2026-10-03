"""The submission check registry.

Its own :class:`Registry`, not the linter's: registering these into the lint collection
would fire them on every lint call and put ``S1.*`` ids into reports that are supposed to
answer a different question. Two registries also make the two analyzers independently
ablatable, which the experiment needs.
"""

from __future__ import annotations

from ...core.check import Registry

SUBMISSION_REGISTRY = Registry("submission")


from .s1_0_not_compilable import *  # noqa: E402,F401,F403  (populates SUBMISSION_REGISTRY)
from .s1_1_no_entry_class import *  # noqa: E402,F401,F403
from .s1_2_entry_has_no_forward import *  # noqa: E402,F401,F403
from .s1_3_unresolved_name import *  # noqa: E402,F401,F403
