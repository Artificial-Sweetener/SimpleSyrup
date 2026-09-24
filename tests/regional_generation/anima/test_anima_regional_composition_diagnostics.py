# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove exact structured diagnostics for regional Anima execution."""

from __future__ import annotations

from uuid import uuid4

import comfy.ops
import pytest
import torch
from comfy.ldm.anima.model import Anima

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
)
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    AnimaActivationGeometry,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_composition_phase import (
    AnimaCompositionPhase,
    AnimaCompositionStage,
)
from simple_syrup.runtime.regional_lora.anima_diagnostics import (
    AnimaRegionalDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import (
    ANIMA_MODULE_SURFACE_DISCOVERY,
    AnimaLoraTargetModule,
    AnimaModuleSurface,
)
from simple_syrup.runtime.regional_lora.anima_targets import (
    ANIMA_BLOCK_COUNT,
    AnimaLoraAdmission,
    AnimaLoraTarget,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget
from simple_syrup.runtime.regional_lora_schedule_resolution import (
    RegionalLoraScheduleResolution,
)

from ..regional.support.regional_lora_test_values import (
    static_lora_schedule,
)


class _RecordingExecutor:
    """Retain a verified class owner and forwarded call arguments."""

    def __init__(self, class_obj: object, result: object) -> None:
        """Store the exact owner and downstream result."""

        self.class_obj = class_obj
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Record one unchanged call and return the configured result."""

        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture(scope="module")
def anima_surface() -> AnimaModuleSurface:
    """Discover a real installed Anima graph with allocation-free meta weights."""

    model = Anima(
        max_img_h=2,
        max_img_w=2,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=2048,
        num_blocks=ANIMA_BLOCK_COUNT,
        num_heads=16,
        mlp_ratio=4.0,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d",
        pos_emb_learnable=False,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=256,
        extra_per_block_abs_pos_emb=False,
        device=torch.device("meta"),
        dtype=torch.float16,
        operations=comfy.ops.disable_weight_init,
    )
    return ANIMA_MODULE_SURFACE_DISCOVERY.discover(model)


def test_zero_pruned_composition_reports_no_low_rank_work(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Distinguish regional attention work from a fully pruned adapter stack."""

    composition = _composition(
        anima_surface,
        torch.zeros((1, 2, 2)),
        include_negative=False,
        all_zero_strength=True,
    )

    snapshot = AnimaRegionalDiagnosticsBuilder(anima_surface, composition).build(
        _geometry(batch=1, height=2, width=2),
        schedule_resolution=static_lora_schedule(composition)[1],
    )
    work = snapshot.work

    assert snapshot.coverage_class == "all_base"
    assert snapshot.active_region_indices == ()
    assert len(snapshot.adapter_uses) == 1
    assert snapshot.adapter_uses[0].active is False
    assert snapshot.adapter_uses[0].pruning_reason == "zero_strength"
    assert work.cross_attention_branch_multiplier == 2.0
    assert work.low_rank_adapter_multiplier == 0.0
    assert work.active_adapter_uses == 0
    assert work.active_target_count == 0
    assert work.target_use_count == 0
    assert work.compatible_projection_batch_count == 0
    assert work.denoiser_call_multiplier == 1.0


def test_schedule_zero_reports_dynamic_pruning_without_hiding_static_support(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Report current inactive adapter work separately from static pruning."""

    composition = _composition(
        anima_surface,
        torch.ones((1, 2, 2)),
        include_negative=False,
    )
    count = len(composition.executions)
    resolution = RegionalLoraScheduleResolution(
        (0.0,) * count,
        (0.0,) * count,
    )

    snapshot = AnimaRegionalDiagnosticsBuilder(anima_surface, composition).build(
        _geometry(batch=1, height=2, width=2),
        schedule_resolution=resolution,
    )

    assert all(not adapter.active for adapter in snapshot.adapter_uses)
    assert all(
        adapter.pruning_reason == "schedule_zero" for adapter in snapshot.adapter_uses
    )
    assert snapshot.work.active_adapter_uses == 0
    assert snapshot.work.active_target_count == 0
    assert snapshot.work.target_use_count == 0


def test_diagnostics_count_every_active_regional_conditioning_entry(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Report entry-branch work after within-region schedule expansion."""

    composition = _composition(
        anima_surface,
        torch.ones((1, 1, 1)),
        include_negative=False,
        regional_entry_count=2,
    )

    snapshot = AnimaRegionalDiagnosticsBuilder(
        anima_surface,
        composition,
    ).build(
        _geometry(batch=1, height=1, width=1),
        schedule_resolution=static_lora_schedule(composition)[1],
    )

    assert snapshot.work.cross_attention_branch_multiplier == 3.0


def test_repeated_chunk_layout_preserves_actual_order_and_latent_slices(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Report repeated reversed CFG chunks and latent batches without assumptions."""

    sequence = (
        RegionalAttentionBranch.NEGATIVE,
        RegionalAttentionBranch.POSITIVE,
        RegionalAttentionBranch.NEGATIVE,
        RegionalAttentionBranch.POSITIVE,
    )
    composition = _composition(
        anima_surface,
        torch.ones((1, 2, 2)),
        include_negative=True,
        branch_sequence=sequence,
        latent_batch_size=2,
    )

    snapshot = AnimaRegionalDiagnosticsBuilder(anima_surface, composition).build(
        _geometry(batch=8, height=2, width=2),
        schedule_resolution=static_lora_schedule(composition)[1],
    )

    assert snapshot.active_branches == ("negative", "positive")
    assert snapshot.positive_chunk_count == 2
    assert snapshot.negative_chunk_count == 2
    assert snapshot.latent_batch_size == 2
    assert [chunk.to_log_fields() for chunk in snapshot.chunks] == [
        {
            "chunk_index": index,
            "branch": branch.value,
            "batch_start": index * 2,
            "batch_stop": (index + 1) * 2,
        }
        for index, branch in enumerate(sequence)
    ]


def _specialization_phase() -> AnimaCompositionPhase:
    """Return stable coordinated phase values for wrapper diagnostics."""

    return AnimaCompositionPhase(
        AnimaCompositionStage.SPECIALIZATION,
        0.4,
        True,
        0.9,
    )


def _composition(
    surface: AnimaModuleSurface,
    masks: torch.Tensor,
    *,
    include_negative: bool,
    all_zero_strength: bool = False,
    branch_sequence: tuple[RegionalAttentionBranch, ...] | None = None,
    latent_batch_size: int = 1,
    regional_entry_count: int = 1,
) -> AnimaRegionalLoraComposition:
    """Build repeated and distinct adapter uses over one installed target path."""

    region_count = int(masks.shape[0])
    sequence = branch_sequence or (
        (
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
        )
        if include_negative
        else (RegionalAttentionBranch.POSITIVE,)
    )
    batch = len(sequence) * latent_batch_size
    context = torch.zeros((batch, 1, 1))
    chunks = tuple(
        RegionalAttentionChunkBatch(
            index,
            branch,
            index * latent_batch_size,
            (index + 1) * latent_batch_size,
        )
        for index, branch in enumerate(sequence)
    )
    attention = AnimaRegionalAttentionExecution(
        BatchedRegionalAttentionContexts(
            latent_batch_size,
            chunks,
            context,
            tuple(
                BatchedRegionalAttentionRegion(
                    region_index,
                    tuple(
                        BatchedRegionalAttentionEntry(
                            entry_index,
                            context.clone(),
                            (1.0,) * batch,
                        )
                        for entry_index in range(regional_entry_count)
                    ),
                )
                for region_index in range(region_count)
            ),
        ),
        RegionalMaskBank(
            masks.clone(),
            masks.clone(),
            int(masks.shape[-1]),
            int(masks.shape[-2]),
        ),
        (1.0,) * region_count,
    )
    descriptor = surface.lora_targets[0]
    first_target = _target(descriptor, scalar=1.0)
    second_target = _target(descriptor, scalar=-0.5)
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    uses: tuple[tuple[AnimaLoraTarget, str, int, float], ...] = (
        (first_target, "private-adapter-path/one.safetensors", 0, 1.0),
        (
            first_target,
            "private-adapter-path/one.safetensors",
            min(1, region_count - 1),
            0.5,
        ),
        (
            second_target,
            "private-adapter-path/two.safetensors",
            min(1, region_count - 1),
            -0.25,
        ),
    )
    if all_zero_strength:
        uses = ((first_target, "zero", 0, 0.0),)
    executions = tuple(
        AnimaRegionalLoraAdapterExecution(
            RegionalLoraAdapterPlan(
                RegionalLoraAdapterIdentity(identity),
                composition_index=index,
                region_index=region,
                branch=(
                    RegionalLoraBranch.NEGATIVE
                    if include_negative and index == 2
                    else RegionalLoraBranch.POSITIVE
                ),
                model_strength=strength,
                schedule=(RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
            ),
            AnimaLoraAdmission((target,)),
            attention,
            model,
            cache,
        )
        for index, (target, identity, region, strength) in enumerate(uses)
    )
    return AnimaRegionalLoraComposition(executions)


def _target(descriptor: AnimaLoraTargetModule, *, scalar: float) -> AnimaLoraTarget:
    """Build one allocation-light rank-one target from a discovered descriptor."""

    tensor = torch.tensor(scalar)
    adapter = StandardLoraTarget(
        descriptor.target_name,
        tensor.expand(1, descriptor.input_features),
        tensor.expand(descriptor.output_features, 1),
        1,
        descriptor.input_features,
        descriptor.output_features,
    )
    return AnimaLoraTarget(descriptor.block_index, descriptor.family, adapter)


def _geometry(
    *,
    batch: int,
    height: int,
    width: int,
    layout: SpatialBatchLayout | None = None,
) -> AnimaActivationGeometry:
    """Build patch-size-one geometry for diagnostics-only tests."""

    return AnimaActivationGeometry(
        batch,
        1,
        height,
        width,
        1,
        1,
        1,
        height,
        width,
        layout,
    )
