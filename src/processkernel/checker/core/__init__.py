"""What both analyzers are built from: the AST front end, the finding vocabulary, and the
check and analyzer base classes.

Nothing analyzer-specific belongs here, and nothing here may import ``torch``, ``triton``
or ``numpy`` -- that is what keeps a whole-run scan a login-node operation at roughly a
millisecond per file.
"""
