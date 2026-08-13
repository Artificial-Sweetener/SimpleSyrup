# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt standard-UNet model derivation to shared lifecycle invariants."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from attention_coupling_invariant_contract import (
    AttentionCouplingLifecycleHarness,
    AttentionCouplingLifecycleObservation,
)
from attention_coupling_invariant_values import scheduled_invariant_plan
from torch import nn

from simple_syrup.runtime.attention_coupling.unet import (
    StandardUnetAttentionBackend,
)
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)


class _UnetModelRoot(nn.Module):
    """Expose a diffusion model at Comfy's standard MODEL path."""

    def __init__(self) -> None:
        """Install one weak-referenceable diffusion module."""

        super().__init__()
        self.diffusion_model = nn.Identity()


class UnetAttentionCouplingLifecycleHarness(AttentionCouplingLifecycleHarness):
    """Derive the production UNet backend and publish common clone facts."""

    def derive(self) -> AttentionCouplingLifecycleObservation:
        """Return one direct clone while proving its source stays unchanged."""

        from comfy.model_patcher import ModelPatcher

        device = torch.device("cpu")
        source = ModelPatcher(
            _UnetModelRoot(),
            load_device=device,
            offload_device=device,
        )
        plan = scheduled_invariant_plan()
        state = StandardUnetAttentionState(
            plan,
            (1.0,),
            RegionalAttentionDiagnosticsBuilder(
                plan.mask_bank,
                backend="invariant.standard-unet",
            ),
        )
        source_options = deepcopy(source.model_options)
        source_objects = source.object_patches.copy()

        built = StandardUnetAttentionBackend().derive(model=source, state=state)
        derived: Any = built.model
        patches = derived.model_options["transformer_options"]["patches"]

        return AttentionCouplingLifecycleObservation(
            derived_is_new=derived is not source,
            direct_parent=derived.parent is source,
            source_unchanged=source.model_options == source_options
            and source.object_patches == source_objects,
            derived_has_mutations=bool(patches["attn2_patch"])
            and bool(patches["attn2_output_patch"]),
            canonical_plan_preserved=built.state.plan is plan
            and built.state.diagnostics.mask_bank is plan.mask_bank,
        )
