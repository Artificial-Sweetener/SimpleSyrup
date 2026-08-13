# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt Anima model derivation to shared lifecycle invariants."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from anima_attention_coupling_lifecycle_values import meta_anima_model
from attention_coupling_invariant_contract import (
    AttentionCouplingLifecycleHarness,
    AttentionCouplingLifecycleObservation,
)
from attention_coupling_invariant_values import scheduled_invariant_plan
from torch import nn

from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.runtime.regional_lora.anima_full_context_backend import (
    FullContextAnimaAttentionBackend,
)
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation


class _AnimaModelRoot(nn.Module):
    """Expose Anima at Comfy's diffusion-model patch root."""

    def __init__(self) -> None:
        """Construct one exact allocation-free installed Anima model."""

        super().__init__()
        self.diffusion_model = meta_anima_model()


class AnimaAttentionCouplingLifecycleHarness(AttentionCouplingLifecycleHarness):
    """Derive the production Anima backend and publish common clone facts."""

    def derive(self) -> AttentionCouplingLifecycleObservation:
        """Return one direct clone while proving its source stays unchanged."""

        from comfy.model_patcher import ModelPatcher

        device = torch.device("cpu")
        source = ModelPatcher(
            _AnimaModelRoot(),
            load_device=device,
            offload_device=device,
        )
        plan = scheduled_invariant_plan()
        source_options = deepcopy(source.model_options)
        source_objects = source.object_patches.copy()
        built = FullContextAnimaAttentionBackend().derive(
            model=source,
            processed_plan=plan,
            adaptation=RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ()),
            region_strengths=(1.0,),
            latent_batch_size=1,
        )
        derived: Any = built.model

        return AttentionCouplingLifecycleObservation(
            derived_is_new=derived is not source,
            direct_parent=derived.parent is source,
            source_unchanged=source.model_options == source_options
            and source.object_patches == source_objects,
            derived_has_mutations=bool(derived.object_patches),
            canonical_plan_preserved=built.attention.mask_bank is plan.mask_bank,
        )
