"""Supply exact UNet identity to shared diagnostic invariant execution."""

from __future__ import annotations

from attention_coupling_diagnostics_harness import build_diagnostics_observation
from attention_coupling_invariant_contract import (
    AttentionCouplingDiagnosticsHarness,
    AttentionCouplingDiagnosticsObservation,
)
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel


class UnetAttentionCouplingDiagnosticsHarness(AttentionCouplingDiagnosticsHarness):
    """Build shared diagnostics under the installed UNet runtime identity."""

    def observe(self) -> AttentionCouplingDiagnosticsObservation:
        """Return one exact shared UNet diagnostic observation."""

        return build_diagnostics_observation(UNetModel)
