# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build shared diagnostic observations for backend-specific test adapters."""

from __future__ import annotations

import torch
from attention_coupling_invariant_contract import (
    AttentionCouplingDiagnosticsObservation,
)
from attention_coupling_invariant_values import scheduled_invariant_plan

from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.regional_attention_diagnostic_values import (
    RegionalAttentionQueryGeometry,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_attention_model_call import (
    RegionalAttentionModelCallResolver,
)


def build_diagnostics_observation(
    runtime_type: type[object],
) -> AttentionCouplingDiagnosticsObservation:
    """Build one shared snapshot using exact backend runtime identity."""

    plan = scheduled_invariant_plan()
    negative = plan.negative.base_context.entries[0].cross_attention
    positive = plan.positive.base_context.entries[0].cross_attention
    contexts = RegionalAttentionModelCallResolver().resolve(
        plan,
        model_input=torch.zeros(2, 4, 1, 1),
        base_context=torch.cat((negative, positive)),
        transformer_options={
            "cond_or_uncond": [1, 0],
            "sigmas": torch.tensor([0.5, 0.5]),
        },
    )
    layout = SpatialBatchLayout(
        canvas_width=1,
        canvas_height=1,
        views=(SpatialView(SpatialViewKind.FULL, 0, 0, 1, 1, 1, 1),),
        input_batch_size=1,
    )
    geometry = RegionalAttentionQueryGeometry(
        input_batch_size=2,
        query_time=1,
        query_height=1,
        query_width=1,
        spatial_layout=None,
    )
    runtime_identity = f"{runtime_type.__module__}.{runtime_type.__qualname__}"
    snapshot = RegionalAttentionDiagnosticsBuilder(
        plan.mask_bank,
        backend=runtime_identity,
    ).build(
        contexts,
        geometry,
        layout,
        transformer_options={"cond_or_uncond": [1, 0]},
    )
    return AttentionCouplingDiagnosticsObservation(
        strategy=snapshot.strategy,
        backend_identity=snapshot.backend,
        runtime_identity=runtime_identity,
        region_count=snapshot.region_count,
        active_region_indices=snapshot.active_region_indices,
        active_branches=snapshot.active_branches,
        cross_attention_branch_multiplier=snapshot.cross_attention_branch_multiplier,
        denoiser_call_multiplier=snapshot.denoiser_call_multiplier,
        query_token_count=snapshot.query_token_count,
    )
