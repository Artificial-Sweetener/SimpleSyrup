# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove exact structured diagnostics for regional Anima execution."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

import comfy.ops
import pytest
import torch
from comfy.ldm.anima.model import Anima
from torch import nn

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
    AnimaActivationContext,
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
from simple_syrup.runtime.regional_lora.anima_composition_phase_context import (
    AnimaCompositionPhaseContext,
)
from simple_syrup.runtime.regional_lora.anima_diagnostics import (
    AnimaRegionalDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_lora.anima_diagnostics_wrapper import (
    AnimaRegionalDiagnosticsDiffusionWrapper,
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


def test_wrapper_emits_one_safe_record_and_preserves_downstream_call(
    anima_surface: AnimaModuleSurface,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit one JSON-safe record without prompt, path, tensor, or output leakage."""

    masks = torch.ones((1, 2, 2))
    composition = _composition(anima_surface, masks, include_negative=False)
    context = AnimaActivationContext()
    phase_context = AnimaCompositionPhaseContext()
    schedule, resolution = static_lora_schedule(composition)
    logger = logging.getLogger("test.anima.regional.diagnostics")
    wrapper = AnimaRegionalDiagnosticsDiffusionWrapper(
        anima_surface,
        context,
        AnimaRegionalDiagnosticsBuilder(anima_surface, composition),
        schedule,
        phase_context,
        logger=logger,
    )
    executor = _RecordingExecutor(anima_surface.diffusion_model, "prediction")
    model_input = torch.zeros((1, 16, 1, 2, 2))
    secret_prompt = "never-log-this-prompt"

    with caplog.at_level(logging.INFO, logger=logger.name):
        with (
            schedule.activate(resolution),
            context.activate(_geometry(batch=1, height=2, width=2)),
            phase_context.activate(_specialization_phase()),
        ):
            result = wrapper(
                executor,
                model_input,
                transformer_options={"prompt": secret_prompt},
            )

    records = [record for record in caplog.records if record.name == logger.name]
    assert result == "prediction"
    assert executor.calls == [
        ((model_input,), {"transformer_options": {"prompt": secret_prompt}})
    ]
    assert len(records) == 1
    assert records[0].__dict__["operation"] == "anima_attention_coupling.execute"
    serialized = json.dumps(records[0].__dict__["regional_diagnostics"], sort_keys=True)
    assert secret_prompt not in serialized
    assert "private-adapter-path" not in serialized
    assert "tensor(" not in serialized
    assert "prediction" not in serialized
    fields = records[0].__dict__["regional_diagnostics"]
    assert fields["composition_phase"] == {
        "stage": "specialization",
        "denoising_progress": 0.4,
        "restrict_self_attention": True,
        "regional_lora_scale": 0.9,
    }


def test_wrapper_validates_without_building_disabled_info_payload(
    anima_surface: AnimaModuleSurface,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid unused snapshot work while retaining dynamic call validation."""

    composition = _composition(
        anima_surface,
        torch.ones((1, 2, 2)),
        include_negative=False,
    )
    context = AnimaActivationContext()
    phase_context = AnimaCompositionPhaseContext()
    schedule, resolution = static_lora_schedule(composition)
    builder = AnimaRegionalDiagnosticsBuilder(anima_surface, composition)
    logger = logging.getLogger("test.anima.regional.diagnostics.disabled")
    logger.setLevel(logging.WARNING)
    wrapper = AnimaRegionalDiagnosticsDiffusionWrapper(
        anima_surface,
        context,
        builder,
        schedule,
        phase_context,
        logger=logger,
    )

    def unexpected_build(*args: object, **kwargs: object) -> object:
        """Fail if a disabled INFO payload is constructed."""

        del args, kwargs
        raise AssertionError("disabled diagnostics payload was built")

    monkeypatch.setattr(builder, "build", unexpected_build)
    executor = _RecordingExecutor(anima_surface.diffusion_model, "prediction")
    model_input = torch.zeros((1, 16, 1, 2, 2))

    with (
        schedule.activate(resolution),
        context.activate(_geometry(batch=1, height=2, width=2)),
    ):
        result = wrapper(executor, model_input, transformer_options={})

    assert result == "prediction"
    assert len(executor.calls) == 1


def test_wrapper_fails_before_logging_for_wrong_model_or_missing_geometry(
    anima_surface: AnimaModuleSurface,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reject invalid wrapper ownership and lifetime without ambiguous records."""

    composition = _composition(
        anima_surface,
        torch.ones((1, 2, 2)),
        include_negative=False,
    )
    context = AnimaActivationContext()
    phase_context = AnimaCompositionPhaseContext()
    schedule, _resolution = static_lora_schedule(composition)
    logger = logging.getLogger("test.anima.regional.diagnostics.failures")
    wrapper = AnimaRegionalDiagnosticsDiffusionWrapper(
        anima_surface,
        context,
        AnimaRegionalDiagnosticsBuilder(anima_surface, composition),
        schedule,
        phase_context,
        logger=logger,
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        with pytest.raises(ValueError, match="does not own"):
            wrapper(_RecordingExecutor(nn.Identity(), None))
        with pytest.raises(RuntimeError, match="unavailable outside"):
            wrapper(_RecordingExecutor(anima_surface.diffusion_model, None))

    assert [record for record in caplog.records if record.name == logger.name] == []


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
