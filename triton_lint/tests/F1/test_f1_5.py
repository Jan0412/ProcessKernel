"""F1.5 nn_module_call."""

from __future__ import annotations

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
