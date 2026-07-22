"""F1.5 nn_module_call."""

from __future__ import annotations

import pytest

from conftest import ELEMENTWISE_KERNEL, src

from triton_lint import build_model


class TestF15NnModuleCall:
    def test_fires_when_module_called(self, check):
        found = check(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.fc = nn.Linear(a, b)
    def forward(self, x):
        return self.fc(x)
"""
            ),
        )
        assert found and found[0].severity == "fail"

    def test_silent_for_weight_holder(self, fired):
        """Holding nn.Linear to own the weights is legitimate -- only calling it is not."""
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.fc = nn.Linear(a, b)
    def forward(self, x):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, self.fc.weight, out, x.numel(), BLOCK=128)
        return out
"""
            ),
        )

    def test_silent_for_dropout(self, fired):
        """nn.Dropout is an identity at eval time: it launches nothing."""
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.1)
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return self.drop(out)
"""
            ),
        )


class TestF15Guards:
    def test_light_module_is_a_warning_not_a_failure(self, check):
        """nn.ReLU does compute something, but it is not where the task's cost lives."""
        found = check(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return self.act(out)
"""
            ),
        )
        assert len(found) == 1
        assert found[0].severity == "warn"
        assert found[0].data["heavy"] == []
        assert "still applies PyTorch modules" in found[0].message

    def test_ignores_calls_to_the_models_own_methods(self, fired):
        """self._run(...) is not an nn module -- only self.<module>(...) counts."""
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + """
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.1)

    def _run(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out

    def forward(self, x, y):
        return self._run(x, y)
"""
            ),
        )


class TestF15AttrTableIsModuleWide:
    """BUG-24: ``nn_modules_in_init`` is keyed on the bare attribute name for the
    whole file, so an ``nn.*`` binding made in one class decides how an identically
    named attribute is judged in every other class.

    The controls below rename the attribute and delete the other class; both make
    the finding vanish, which pins the bug to the shared attribute name rather than
    to anything the flagged forward() does.
    """

    HOLDER = '''
class Conv3x3Triton(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3)          # weight holder -- never called

    def forward(self, x):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, self.conv.weight, out, x.numel(), BLOCK=128)
        return out


class ModelNew(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = Conv3x3Triton(cin, cout)

    def forward(self, x):
        return self.conv(x)
'''

    def test_local_submodule_is_not_an_nn_call_when_it_shares_an_attr_name(self, fired):
        assert not fired("F1.5", src(ELEMENTWISE_KERNEL + self.HOLDER))

    def test_control_same_file_with_the_holder_attr_renamed_is_silent(self, fired):
        """The only edit is `self.conv` -> `self.inner` inside the holder."""
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + self.HOLDER.replace(
                    "self.conv = nn.Conv2d(cin, cout, 3)", "self.inner = nn.Conv2d(cin, cout, 3)"
                ).replace("self.conv.weight", "self.inner.weight")
            ),
        )

    def test_control_calling_the_nn_module_really_does_fire(self, check):
        """The weight holder becomes a genuine fallback the moment it is called."""
        found = check(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + self.HOLDER.replace(
                    "        out = torch.empty_like(x)\n"
                    "        add_kernel[(1,)](x, self.conv.weight, out, x.numel(), BLOCK=128)\n"
                    "        return out",
                    "        return self.conv(x)",
                )
            ),
        )
        assert [f.severity for f in found] == ["fail"]

    def test_dead_reference_class_is_not_scanned_when_forward_is_inherited(self, fired):
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + '''
class Reference(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(n, n)

    def forward(self, x):
        return self.fc(x)


class Base(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, x.numel(), BLOCK=128)
        return out


class ModelNew(Base):
    pass
'''
            ),
        )

    def test_nn_parameter_is_not_a_module(self, fired):
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + '''
class EqualizedWeight(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n))

    def forward(self):
        return self.weight * 2.0


class ModelNew(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = EqualizedWeight(n)

    def forward(self, x):
        w = self.weight()
        out = torch.empty_like(x)
        add_kernel[(1,)](x, w, out, x.numel(), BLOCK=128)
        return out
'''
            ),
        )


class TestF15SequentialOfLocalModules:
    """BUG-30: F1.5 treats any constructed-and-called ``nn.*`` module as a torch
    fallback, but a container (``nn.Sequential`` / ``nn.ModuleList``) that wraps the
    file's *own* Triton modules invokes only Triton kernels. Unlike F1.4 -- which grew
    ``_is_local_call`` to stop grading model-authored submodules as fallbacks
    (BUG-16 / BUG-24) -- F1.5 never consults ``local_classes`` / ``attr_classes``, so
    the container lands in ``nn_modules_in_init`` as ``nn.Sequential`` and forward
    calling it is reported at fail. The advice ("keep it only as a weight holder, do
    not call it") is nonsense for a Sequential of compute modules: it owns no weights,
    and not calling it skips the kernels. Synthetic -- the run's Sequential findings
    were torch-layer stacks (real) or externally-supplied ``fn`` (ambiguous)."""

    _LOCAL_TRITON_BLOCK = '''
class TritonReLU(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, x, out, x.numel(), BLOCK=128)
        return out
'''

    def test_sequential_of_local_triton_modules_is_not_a_fallback(self, fired):
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + self._LOCAL_TRITON_BLOCK
                + '''
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(TritonReLU(), TritonReLU())

    def forward(self, x):
        return self.net(x)
'''
            ),
        )

    def test_sequential_of_torch_layers_still_fires(self, check):
        # Passing control: a Sequential of real torch layers IS a fallback and must
        # keep firing -- the fix must inspect the container's contents, not mute it.
        found = check(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + '''
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 8), nn.ReLU())

    def forward(self, x):
        return self.net(x)
'''
            ),
        )
        assert [f.severity for f in found] == ["fail"]

    def test_directly_held_local_module_is_already_silent(self, fired):
        # Passing control: held without the nn.Sequential wrapper, the same local
        # Triton module is correctly not flagged -- pinning the bug to the container.
        assert not fired(
            "F1.5",
            src(
                ELEMENTWISE_KERNEL
                + self._LOCAL_TRITON_BLOCK
                + '''
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = TritonReLU()

    def forward(self, x):
        return self.block(x)
'''
            ),
        )

    _SEQ_COMPREHENSION = '''
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*[TritonReLU() for _ in range(3)])

    def forward(self, x):
        return self.net(x)
'''

    def test_sequential_built_by_unpacking_local_modules_is_not_a_fallback(self, fired):
        assert not fired(
            "F1.5", src(ELEMENTWISE_KERNEL + self._LOCAL_TRITON_BLOCK + self._SEQ_COMPREHENSION)
        )

    def test_control_the_linter_resolves_the_sequential_to_local_classes(self):
        """The linter *has* what a fix needs: `attr_classes['net']` records that `net`
        wraps the file-defined TritonReLU. F1.5 simply never consults it (unlike F1.4's
        `_is_local_call`). Mirrors the F1.4 BUG-24 control -- premise for the fix."""
        model = build_model(
            src(ELEMENTWISE_KERNEL + self._LOCAL_TRITON_BLOCK + self._SEQ_COMPREHENSION), "<t>"
        )
        assert model.attr_classes["net"] == ["TritonReLU"]
        assert model.nn_modules_in_init["ModelNew"]["net"] == "nn.Sequential"


# ---------------------------------------------------------------------------
# Fix-hardening regression tests (Phase 1): pin the false negatives the fail-safe
# rules were designed to avoid. See tests/BUGS.md (BUG-24, BUG-30).
# ---------------------------------------------------------------------------

_LOCAL_TRITON_RELU = '''
class TritonReLU(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, x, out, x.numel(), BLOCK=128)
        return out
'''


def test_mixed_sequential_of_local_and_torch_modules_still_fires(check):
    # BUG-30 boundary: a Sequential that wraps a local Triton module AND a real nn.Linear
    # genuinely invokes torch (cuDNN/gemm) -- the container guard must NOT mute it.
    found = check(
        "F1.5",
        src(
            ELEMENTWISE_KERNEL
            + _LOCAL_TRITON_RELU
            + '''
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(TritonReLU(), nn.Linear(8, 8))

    def forward(self, x):
        return self.net(x)
'''
        ),
    )
    assert [f.severity for f in found] == ["fail"]


def test_child_binds_nn_module_and_inherits_forward_still_fires(check):
    # BUG-24 MRO: the nn.Linear binding lives in the child's __init__ while the forward that
    # calls self.fc is inherited from Base -- the per-class lookup must resolve the binding
    # through the constructed class, or a genuine fallback is silently muted.
    found = check(
        "F1.5",
        src(
            ELEMENTWISE_KERNEL
            + '''
class Base(nn.Module):
    def forward(self, x):
        return self.fc(x)


class ModelNew(Base):
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(n, n)
'''
        ),
    )
    assert [f.severity for f in found] == ["fail"]
