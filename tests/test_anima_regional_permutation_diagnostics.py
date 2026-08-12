# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deliberately corrupt one Anima ordering axis and require diagnostics."""

from __future__ import annotations

import json
from uuid import uuid4

import comfy.ops
import pytest
import torch
from comfy.ldm.anima.model import Anima
from regional_attention_test_values import single_entry_regions
from regional_lora_test_values import static_lora_schedule

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
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
    SpatialView,
    SpatialViewKind,
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
from simple_syrup.runtime.regional_lora.anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)
from simple_syrup.runtime.regional_lora.anima_diagnostic_values import (
    AnimaRegionalExecutionDiagnostics,
)
from simple_syrup.runtime.regional_lora.anima_diagnostics import (
    AnimaRegionalDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocationContext,
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_linear_mutations import (
    anima_lora_composition_linear_mutations,
)
from simple_syrup.runtime.regional_lora.anima_lora_weights import (
    AnimaLoraWeightResolver,
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
    AnimaLoraTargetFamily,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget
from simple_syrup.runtime.spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


@pytest.fixture(scope="module")
def anima_surface() -> AnimaModuleSurface:
    """Discover a real installed Anima graph with allocation-free meta weights."""

    return ANIMA_MODULE_SURFACE_DISCOVERY.discover(
        Anima(
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
    )


def test_cond_or_uncond_permutation_fails_with_expected_and_observed_order(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Reject a reversed selector list against immutable aligned chunk order."""

    target = _target(anima_surface.lora_targets[0], scalar=1.0)
    composition = _composition(
        (target,),
        identities=("adapter-a",),
        regions=(0,),
        branches=(
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
        ),
    )
    builder = AnimaRegionalDiagnosticsBuilder(anima_surface, composition)

    with pytest.raises(
        ValueError,
        match=r"cond_or_uncond order mismatch.*expected.*\(0, 1\).*observed.*\(1, 0\)",
    ):
        builder.build(
            _geometry(batch=2),
            transformer_options={"cond_or_uncond": [1, 0]},
            schedule_resolution=static_lora_schedule(composition)[1],
        )

    valid = builder.build(
        _geometry(batch=2),
        transformer_options={"cond_or_uncond": [0, 1]},
        schedule_resolution=static_lora_schedule(composition)[1],
    )
    assert valid.active_branches == ("positive", "negative")


def test_batch_and_mask_layout_permutations_fail_before_snapshot(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Reject truncated batches and layouts owning the wrong source batch."""

    target = _target(anima_surface.lora_targets[0], scalar=1.0)
    composition = _composition(
        (target,),
        identities=("adapter-a",),
        regions=(0,),
        branches=(
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.NEGATIVE,
        ),
    )
    builder = AnimaRegionalDiagnosticsBuilder(anima_surface, composition)

    with pytest.raises(
        ValueError,
        match="batch-layout mismatch: expected view-major batch 2, observed 1",
    ):
        builder.build(
            _geometry(batch=1),
            schedule_resolution=static_lora_schedule(composition)[1],
        )

    wrong_layout = _tile_layout(input_batch_size=2)
    with pytest.raises(
        ValueError,
        match="mask-layout mismatch.*expanded batch 4.*context batch 2",
    ):
        builder.build(
            _geometry(batch=4, layout=wrong_layout),
            schedule_resolution=static_lora_schedule(composition)[1],
        )


def test_published_layout_identity_and_view_order_are_diagnostic(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Reject an equal replacement layout and expose reversed view order exactly."""

    target = _target(anima_surface.lora_targets[0], scalar=1.0)
    composition = _composition(
        (target,),
        identities=("adapter-a",),
        regions=(0,),
        branches=(
            RegionalAttentionBranch.POSITIVE,
            RegionalAttentionBranch.POSITIVE,
        ),
    )
    builder = AnimaRegionalDiagnosticsBuilder(anima_surface, composition)
    authoritative = _tile_layout(input_batch_size=1)
    equal_replacement = _tile_layout(input_batch_size=1)

    with pytest.raises(ValueError, match="published spatial layout does not match"):
        builder.build(
            _geometry(batch=2, layout=authoritative),
            transformer_options={
                "cond_or_uncond": [0, 0],
                SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {
                    SPATIAL_BATCH_LAYOUT_KEY: equal_replacement
                },
            },
            schedule_resolution=static_lora_schedule(composition)[1],
        )

    baseline = builder.build(
        _geometry(batch=2, layout=authoritative),
        transformer_options={
            "cond_or_uncond": [0, 0],
            SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {
                SPATIAL_BATCH_LAYOUT_KEY: authoritative
            },
        },
        schedule_resolution=static_lora_schedule(composition)[1],
    )
    reversed_layout = SpatialBatchLayout(
        2,
        1,
        tuple(reversed(authoritative.views)),
        1,
    )
    permuted = builder.build(
        _geometry(batch=2, layout=reversed_layout),
        schedule_resolution=static_lora_schedule(composition)[1],
    )
    with pytest.raises(ValueError, match="view-order permutation diagnostic"):
        _require_same_diagnostic_axis(
            "view-order",
            tuple(view.to_log_fields() for view in baseline.views),
            tuple(view.to_log_fields() for view in permuted.views),
        )


def test_region_and_adapter_order_permutations_change_safe_diagnostics(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Expose region ownership and declared adapter order without source paths."""

    descriptor = anima_surface.lora_targets[0]
    first = _target(descriptor, scalar=1.0)
    second = _target(descriptor, scalar=-0.5)
    baseline = _snapshot(
        anima_surface,
        targets=(first, second),
        identities=(
            r"<FIXTURE_ROOT>\adapter-a.safetensors",
            r"<FIXTURE_ROOT>\adapter-b.safetensors",
        ),
        regions=(0, 1),
    )
    region_permuted = _snapshot(
        anima_surface,
        targets=(first, second),
        identities=(
            r"<FIXTURE_ROOT>\adapter-a.safetensors",
            r"<FIXTURE_ROOT>\adapter-b.safetensors",
        ),
        regions=(1, 0),
    )
    adapter_permuted = _snapshot(
        anima_surface,
        targets=(second, first),
        identities=(
            r"<FIXTURE_ROOT>\adapter-b.safetensors",
            r"<FIXTURE_ROOT>\adapter-a.safetensors",
        ),
        regions=(1, 0),
    )

    with pytest.raises(ValueError, match="region-order permutation diagnostic"):
        _require_same_diagnostic_axis(
            "region-order",
            tuple(use.region_index for use in baseline.adapter_uses),
            tuple(use.region_index for use in region_permuted.adapter_uses),
        )
    with pytest.raises(ValueError, match="adapter-order permutation diagnostic"):
        _require_same_diagnostic_axis(
            "adapter-order",
            tuple(use.adapter_token for use in baseline.adapter_uses),
            tuple(use.adapter_token for use in adapter_permuted.adapter_uses),
        )
    serialized = json.dumps(baseline.to_log_fields())
    assert r"<FIXTURE_ROOT>" not in serialized
    assert [use.composition_index for use in baseline.adapter_uses] == [0, 1]


@pytest.mark.parametrize("corruption", ["unknown_key", "wrong_descriptor"])
def test_target_key_permutation_fails_against_verified_surface(
    anima_surface: AnimaModuleSurface,
    corruption: str,
) -> None:
    """Reject unknown keys and key/family descriptor disagreement before patching."""

    descriptor = anima_surface.lora_targets[0]
    if corruption == "unknown_key":
        target = _target(
            descriptor,
            scalar=1.0,
            target_name="diffusion_model.blocks.0.permuted_target",
        )
        message = "target-key mismatch.*permuted_target.*absent"
    else:
        target = _target(
            descriptor,
            scalar=1.0,
            family=AnimaLoraTargetFamily.CROSS_ATTN_Q,
        )
        message = "target descriptor mismatch.*expected block.*observed block"
    composition = _composition(
        (target,),
        identities=("adapter-a",),
        regions=(0,),
        branches=(RegionalAttentionBranch.POSITIVE,),
    )

    with pytest.raises(ValueError, match=message):
        anima_lora_composition_linear_mutations(
            anima_surface,
            composition,
            weight_resolver=AnimaLoraWeightResolver(
                cross_attention_context=AnimaCrossAttentionInvocationContext(),
                branch_context=AnimaLoraBranchInvocationContext(),
                spatial_context=AnimaLoraSpatialInvocationContext(),
            ),
            schedule_context=static_lora_schedule(composition)[0],
        )


def _snapshot(
    surface: AnimaModuleSurface,
    *,
    targets: tuple[AnimaLoraTarget, ...],
    identities: tuple[str, ...],
    regions: tuple[int, ...],
) -> AnimaRegionalExecutionDiagnostics:
    """Build one diagnostic snapshot for order-sensitive test values."""

    composition = _composition(
        targets,
        identities=identities,
        regions=regions,
        branches=(RegionalAttentionBranch.POSITIVE,),
    )
    return AnimaRegionalDiagnosticsBuilder(surface, composition).build(
        _geometry(batch=1),
        schedule_resolution=static_lora_schedule(composition)[1],
    )


def _composition(
    targets: tuple[AnimaLoraTarget, ...],
    *,
    identities: tuple[str, ...],
    regions: tuple[int, ...],
    branches: tuple[RegionalAttentionBranch, ...],
    latent_batch_size: int = 1,
) -> AnimaRegionalLoraComposition:
    """Build ordered adapter uses over a hard two-region canonical mask bank."""

    if not (len(targets) == len(identities) == len(regions)):
        raise AssertionError("Permutation fixture adapter axes must align.")
    context = torch.zeros((len(branches) * latent_batch_size, 1, 1))
    chunks = tuple(
        RegionalAttentionChunkBatch(
            index,
            branch,
            index * latent_batch_size,
            (index + 1) * latent_batch_size,
        )
        for index, branch in enumerate(branches)
    )
    masks = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    attention = AnimaRegionalAttentionExecution(
        BatchedRegionalAttentionContexts(
            latent_batch_size,
            chunks,
            context,
            single_entry_regions((context.clone(), context.clone())),
        ),
        RegionalMaskBank(masks.clone(), masks.clone(), 2, 1),
        (1.0, 1.0),
    )
    model = ModelCloneLineage(uuid4(), uuid4())
    cache = RegionalLoraExecutionCache()
    executions = tuple(
        AnimaRegionalLoraAdapterExecution(
            RegionalLoraAdapterPlan(
                RegionalLoraAdapterIdentity(identity),
                composition_index=index,
                region_index=region,
                branch=RegionalLoraBranch.POSITIVE,
                model_strength=1.0,
                schedule=(RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
            ),
            AnimaLoraAdmission((target,)),
            attention,
            model,
            cache,
        )
        for index, (target, identity, region) in enumerate(
            zip(targets, identities, regions, strict=True)
        )
    )
    return AnimaRegionalLoraComposition(executions)


def _target(
    descriptor: AnimaLoraTargetModule,
    *,
    scalar: float,
    target_name: str | None = None,
    family: AnimaLoraTargetFamily | None = None,
) -> AnimaLoraTarget:
    """Build one allocation-light target with optional key/family corruption."""

    value = torch.tensor(scalar)
    adapter = StandardLoraTarget(
        target_name or descriptor.target_name,
        value.expand(1, descriptor.input_features),
        value.expand(descriptor.output_features, 1),
        1,
        descriptor.input_features,
        descriptor.output_features,
    )
    return AnimaLoraTarget(
        descriptor.block_index,
        descriptor.family if family is None else family,
        adapter,
    )


def _tile_layout(*, input_batch_size: int) -> SpatialBatchLayout:
    """Return two ordered one-pixel views over a two-pixel canvas."""

    return SpatialBatchLayout(
        2,
        1,
        (
            SpatialView(SpatialViewKind.TILE, 0, 0, 1, 1, 1, 1),
            SpatialView(SpatialViewKind.TILE, 1, 0, 1, 1, 1, 1),
        ),
        input_batch_size,
    )


def _geometry(
    *,
    batch: int,
    layout: SpatialBatchLayout | None = None,
) -> AnimaActivationGeometry:
    """Build a one-row patch-size-one activation geometry."""

    width = 2 if layout is None else 1
    return AnimaActivationGeometry(batch, 1, 1, width, 1, 1, 1, 1, width, layout)


def _require_same_diagnostic_axis(
    axis: str,
    expected: object,
    observed: object,
) -> None:
    """Raise an actionable mismatch when one diagnostic axis was permuted."""

    if observed != expected:
        raise ValueError(
            f"Anima {axis} permutation diagnostic mismatch: expected "
            f"{expected!r}, observed {observed!r}."
        )
