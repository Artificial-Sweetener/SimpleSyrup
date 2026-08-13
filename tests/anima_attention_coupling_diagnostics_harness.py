# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Supply exact Anima identity to shared diagnostic invariant execution."""

from __future__ import annotations

from attention_coupling_diagnostics_harness import build_diagnostics_observation
from attention_coupling_invariant_contract import (
    AttentionCouplingDiagnosticsHarness,
    AttentionCouplingDiagnosticsObservation,
)
from comfy.ldm.anima.model import Anima


class AnimaAttentionCouplingDiagnosticsHarness(AttentionCouplingDiagnosticsHarness):
    """Build shared diagnostics under the installed Anima runtime identity."""

    def observe(self) -> AttentionCouplingDiagnosticsObservation:
        """Return one exact shared Anima diagnostic observation."""

        return build_diagnostics_observation(Anima)
