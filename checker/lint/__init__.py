"""The Triton linter: "is this good Triton?"

Family 1 asks whether the kernel is real -- whether the work it claims to do on the GPU is
actually reachable and not quietly delegated back to torch. Family 2 asks what it wastes
once it is. Both run over the model :mod:`checker.core.parsing` builds, extended here by
the kernel-body, host-flow and shape stages.
"""
