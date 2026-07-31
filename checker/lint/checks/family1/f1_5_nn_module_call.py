"""F1.5 -- the solution calls a PyTorch nn module to do the work.

The module-level form of F1.4: ``self.conv = nn.Conv2d(...)`` in ``__init__`` and
``self.conv(x)`` in ``forward`` means cuDNN computed the answer.

Two false-positive guards, both of which matter:

1. **Weight holders.** Holding an ``nn.Conv2d`` purely to *own the weights* is
   completely legitimate::

       self.conv = nn.Conv2d(...)                            # fine: weight holder
       conv_kernel[grid](x, self.conv.weight, out, ...)      # fine

   What is not legitimate is *invoking* it. So the check is "is a constructed nn
   module ever called", not "is one constructed". Attribute reads (``.weight``,
   ``.bias``) are not calls and never reach ``host_calls``, so this guard is
   structural.

2. **Modules that compute nothing.** ``nn.Dropout`` is an identity at eval time and
   launches no kernel at all; ``nn.Identity`` and ``nn.Flatten`` are free. Flagging
   these is noise, and the "keep it as a weight holder" advice would be nonsense for
   them.

References: same as F1.4 -- AutoTriton (arXiv:2507.05687), TritonRL (arXiv:2510.17891).
"""

from __future__ import annotations

from ...hostflow import NN_CONTAINERS
from ....core.check import Check
from ....core.model import Finding, ModuleModel
from .. import LINT_REGISTRY
from .f1_4_torch_fallback import _host_scopes, _nn_binding

#: Launch no kernel (or are an identity) at inference -- calling them is not cheating.
INERT_MODULES = {
    "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout",
    "FeatureAlphaDropout", "Identity", "Flatten", "Unflatten",
}

#: Modules that carry the task's real computation.
HEAVY_MODULES = {
    "Linear", "LazyLinear", "Bilinear", "Embedding", "EmbeddingBag",
    "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "GroupNorm",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "RMSNorm",
    "MultiheadAttention", "Transformer", "TransformerEncoderLayer",
    "LSTM", "GRU", "RNN",
    "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "Softmax", "LogSoftmax", "Sequential",
}


def _module_name(cls: str) -> str:
    return cls.rsplit(".", 1)[-1]


@LINT_REGISTRY.add
class NnModuleCall(Check):
    check_id = "F1.5"
    name = "nn_module_call"
    severity = "fail"

    def run(self, model: ModuleModel) -> list[Finding]:
        if not model.nn_modules_in_init:
            return []

        scopes = _host_scopes(model)
        heavy: list[tuple[str, str, int]] = []
        light: list[tuple[str, str, int]] = []

        for call in model.host_calls:
            if call.enclosing not in scopes or not call.qualname.startswith("self."):
                continue
            attr = call.qualname.split(".", 1)[1]
            cls = _nn_binding(model, call.enclosing.rsplit(".", 1)[0], attr)
            if not cls:
                continue  # not an nn.* binding in this class -- not a fallback (BUG-24)

            name = _module_name(cls)
            # BUG-30: a container (Sequential/ModuleList) that wraps only the file's own Triton
            # modules invokes only kernels. It owns no weights and skipping the call would skip
            # the kernels, so the "keep it as a weight holder" advice is nonsense. A container
            # that also builds a real nn module (`containers_with_torch`) is still a fallback.
            if (
                name in NN_CONTAINERS
                and attr in model.attr_classes
                and attr not in model.containers_with_torch
            ):
                continue
            if name in INERT_MODULES:
                continue  # a no-op at inference: launches nothing, computes nothing
            if name in HEAVY_MODULES:
                heavy.append((attr, cls, call.lineno))
            else:
                light.append((attr, cls, call.lineno))

        hits = heavy + light
        if not hits:
            return []

        listed = ", ".join(f"`self.{a}` ({c}, line {ln})" for a, c, ln in hits)
        if heavy:
            message = (
                f"forward() invokes PyTorch modules to do the computation: {listed}. "
                f"Keep the module only as a weight holder (read `.weight` / `.bias` and pass "
                f"them to your Triton kernel) -- do not call it."
            )
        else:
            message = (
                f"forward() still applies PyTorch modules: {listed}. Fold their computation "
                f"into your Triton kernel instead of calling them."
            )

        return [
            self.finding(
                message,
                severity="fail" if heavy else "warn",
                modules=[{"attr": a, "cls": c, "lineno": ln} for a, c, ln in hits],
                heavy=[c for _, c, _ in heavy],
                lineno=min(ln for _, _, ln in hits),
            )
        ]


