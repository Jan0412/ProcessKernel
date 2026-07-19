"""F1.5 nn_module_call."""

from __future__ import annotations

import pytest

from conftest import ELEMENTWISE_KERNEL, src


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

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-24: Conv3x3Triton holds nn.Conv2d purely to own the weights and "
        "passes .weight to its kernel -- exactly what F1.5's own advice prescribes and "
        "what its docstring calls a legitimate weight holder. But `conv` lands in the "
        "module-wide nn_modules_in_init, so ModelNew.forward calling its own "
        "Conv3x3Triton is reported at fail as invoking nn.Conv2d (real sample: p2179_s3)",
    )
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

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-27: F1.5 shares _host_scopes with F1.4, so it inherits the same "
        "all-functions fallback. ModelNew inherits its forward -> entry is None -> the "
        "dead Reference class is scanned and its nn.Linear call reported at fail as "
        "something the timed forward does",
    )
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

    @pytest.mark.xfail(
        strict=True,
        reason="BUG-24: nn_modules_in_init is filled by a `nn.`/`torch.nn.` namespace "
        "prefix test, which admits things that are not modules at all. nn.Parameter is "
        "a tensor: it launches nothing and cannot be called. Reported as a PyTorch "
        "module the forward should fold into a kernel (real sample: p16312_s1)",
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
