# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove complete installed-Anima mutation composition and restoration."""

from __future__ import annotations

from typing import Any

import comfy.ops
import pytest
import torch
from comfy.ldm.anima.model import Anima
from comfy.ldm.cosmos.predict2 import Attention, Block, GPT2FeedForward
from comfy.patcher_extension import CallbacksMP
from regional_attention_test_values import single_entry_regions
from torch import nn

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
from simple_syrup.runtime.model_patcher_mutations import ModelKeyedCallbackMutation
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    ANIMA_ACTIVATION_WRAPPER_KEY,
    AnimaActivationDiffusionWrapper,
)
from simple_syrup.runtime.regional_lora.anima_attention_coupling import (
    anima_attention_coupling_mutations,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_block_execution import (
    AnimaRegionalLoraBlockPatch,
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
from simple_syrup.runtime.regional_lora.anima_composition_phase_wrapper import (
    ANIMA_COMPOSITION_PHASE_WRAPPER_KEY,
    AnimaCompositionPhaseDiffusionWrapper,
)
from simple_syrup.runtime.regional_lora.anima_cross_attention import (
    AnimaRegionalCrossAttentionPatch,
)
from simple_syrup.runtime.regional_lora.anima_diagnostics_wrapper import (
    ANIMA_DIAGNOSTICS_WRAPPER_KEY,
    AnimaRegionalDiagnosticsDiffusionWrapper,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_linear_execution import (
    AnimaRegionalLoraCompositionLinearPatch,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import (
    ANIMA_MODULE_SURFACE_DISCOVERY,
    AnimaModuleSurface,
)
from simple_syrup.runtime.regional_lora.anima_schedule_context import (
    AnimaRegionalLoraScheduleContext,
)
from simple_syrup.runtime.regional_lora.anima_schedule_wrapper import (
    ANIMA_REGIONAL_LORA_SCHEDULE_WRAPPER_KEY,
    AnimaRegionalLoraScheduleDiffusionWrapper,
)
from simple_syrup.runtime.regional_lora.anima_self_attention import (
    AnimaRegionalSelfAttentionPatch,
)
from simple_syrup.runtime.regional_lora.anima_targets import (
    ANIMA_BLOCK_COUNT,
    AnimaLoraAdmission,
    AnimaLoraTarget,
    AnimaLoraTargetFamily,
    anima_lora_target_name,
    expected_anima_lora_features,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget


class _AnimaModelRoot(nn.Module):
    """Expose installed Anima under Comfy's object-patch root path."""

    def __init__(self, diffusion_model: nn.Module) -> None:
        """Retain the exact diffusion model."""

        super().__init__()
        self.diffusion_model = diffusion_model


class _ScheduleRecordingExecutor:
    """Record schedule state visible inside one nested model call."""

    def __init__(
        self,
        class_obj: object,
        context: AnimaRegionalLoraScheduleContext,
    ) -> None:
        """Retain the exact model and task-local context."""

        self.class_obj = class_obj
        self._context = context
        self.strengths: list[tuple[float, ...]] = []

    def __call__(self, *args: object, **kwargs: object) -> str:
        """Record active strengths and preserve arbitrary arguments."""

        del args, kwargs
        self.strengths.append(self._context.require_current().effective_strengths)
        return "prediction"


class _FullLoraPhaseContext(AnimaCompositionPhaseContext):
    """Publish full LoRA scale for focused schedule-wrapper tests."""

    def require_current(self) -> AnimaCompositionPhase:
        """Return one stable specialization phase."""

        return AnimaCompositionPhase(
            AnimaCompositionStage.SPECIALIZATION, 0.5, True, 1.0
        )


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


def test_attention_only_composition_excludes_every_lora_mutation(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Install cross-attention coupling without LoRA block or schedule work."""

    source = _patcher(_AnimaModelRoot(anima_surface.diffusion_model))
    attention = _composition(source, _admission()).attention
    mutations = anima_attention_coupling_mutations(anima_surface, attention)

    assert (len(mutations), getattr(mutations[2], "key", None)) == (
        3 + (2 * ANIMA_BLOCK_COUNT) + 2,
        "simple_syrup.anima_attention_diagnostics",
    )
    attention_cache_lifecycle = mutations[-2]
    assert isinstance(attention_cache_lifecycle, ModelKeyedCallbackMutation)
    assert attention_cache_lifecycle.callback_type == CallbacksMP.ON_DETACH
    assert attention_cache_lifecycle.key == "simple_syrup.anima_attention_device_cache"
    partition_cache_lifecycle = mutations[-1]
    assert isinstance(partition_cache_lifecycle, ModelKeyedCallbackMutation)
    assert partition_cache_lifecycle.key == (
        "simple_syrup.anima_self_attention_partition_cache"
    )
    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        mutations,
        operation="attention-only Anima coupling",
    )
    assert len(derived.object_patches) == ANIMA_BLOCK_COUNT * 2
    assert len(derived.get_all_wrappers("diffusion_model")) == 3
    assert (
        len(derived.get_wrappers("diffusion_model", ANIMA_ACTIVATION_WRAPPER_KEY)) == 1
    )

    derived.patch_model(load_weights=False)
    try:
        for block_index, block_surface in enumerate(anima_surface.blocks):
            active_block = anima_surface.diffusion_model.blocks[block_index]
            assert active_block is block_surface.block
            assert isinstance(active_block.self_attn, AnimaRegionalSelfAttentionPatch)
            assert isinstance(active_block.cross_attn, AnimaRegionalCrossAttentionPatch)
    finally:
        derived.unpatch_model(unpatch_weights=False)

    for block_index, block_surface in enumerate(anima_surface.blocks):
        active_block = anima_surface.diffusion_model.blocks[block_index]
        assert active_block is block_surface.block
        assert active_block.self_attn is block_surface.self_attention
        assert active_block.cross_attn is block_surface.cross_attention


def test_composition_installs_and_restores_all_448_targets(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Round-trip every clone-local wrapper, block, attention, and linear owner."""

    source = _patcher(_AnimaModelRoot(anima_surface.diffusion_model))
    composition = _composition(source, _admission())
    mutations = anima_attention_coupling_mutations(
        anima_surface,
        composition.attention,
        composition=composition,
    )
    class_forwards = (
        Anima.forward,
        Block.forward,
        Attention.forward,
        GPT2FeedForward.forward,
    )

    assert len(mutations) == 4 + (3 * ANIMA_BLOCK_COUNT) + 448 + 4
    paths = tuple(getattr(mutation, "path", None) for mutation in mutations[4:-4])
    assert paths[:ANIMA_BLOCK_COUNT] == tuple(
        f"diffusion_model.blocks.{index}" for index in range(ANIMA_BLOCK_COUNT)
    )
    assert paths[ANIMA_BLOCK_COUNT : 2 * ANIMA_BLOCK_COUNT] == tuple(
        f"diffusion_model.blocks.{index}.self_attn"
        for index in range(ANIMA_BLOCK_COUNT)
    )
    assert paths[2 * ANIMA_BLOCK_COUNT : 3 * ANIMA_BLOCK_COUNT] == tuple(
        f"diffusion_model.blocks.{index}.cross_attn"
        for index in range(ANIMA_BLOCK_COUNT)
    )
    assert paths[3 * ANIMA_BLOCK_COUNT :] == tuple(
        anima_lora_target_name(block_index, family)
        for block_index in range(ANIMA_BLOCK_COUNT)
        for family in AnimaLoraTargetFamily
    )
    cache_lifecycle = mutations[-4]
    assert isinstance(cache_lifecycle, ModelKeyedCallbackMutation)
    assert cache_lifecycle.callback_type == CallbacksMP.ON_DETACH
    assert cache_lifecycle.key == "simple_syrup.anima_regional_lora_device_cache"
    attention_cache_lifecycle = mutations[-3]
    assert isinstance(attention_cache_lifecycle, ModelKeyedCallbackMutation)
    assert attention_cache_lifecycle.callback_type == CallbacksMP.ON_DETACH
    assert attention_cache_lifecycle.key == "simple_syrup.anima_attention_device_cache"
    schedule_cache_lifecycle = mutations[-2]
    assert isinstance(schedule_cache_lifecycle, ModelKeyedCallbackMutation)
    assert schedule_cache_lifecycle.callback_type == CallbacksMP.ON_DETACH
    assert (
        schedule_cache_lifecycle.key
        == "simple_syrup.anima_regional_lora_schedule_session"
    )
    partition_cache_lifecycle = mutations[-1]
    assert isinstance(partition_cache_lifecycle, ModelKeyedCallbackMutation)
    assert partition_cache_lifecycle.key == (
        "simple_syrup.anima_self_attention_partition_cache"
    )

    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        mutations,
        operation="complete regional Anima adapter composition",
    )
    assert source.object_patches == {}
    assert len(derived.object_patches) == ANIMA_BLOCK_COUNT * 19
    activation_wrappers = derived.get_wrappers(
        "diffusion_model", ANIMA_ACTIVATION_WRAPPER_KEY
    )
    diagnostic_wrappers = derived.get_wrappers(
        "diffusion_model", ANIMA_DIAGNOSTICS_WRAPPER_KEY
    )
    schedule_wrappers = derived.get_wrappers(
        "diffusion_model", ANIMA_REGIONAL_LORA_SCHEDULE_WRAPPER_KEY
    )
    phase_wrappers = derived.get_wrappers(
        "diffusion_model", ANIMA_COMPOSITION_PHASE_WRAPPER_KEY
    )
    assert (
        len(schedule_wrappers)
        == len(phase_wrappers)
        == len(activation_wrappers)
        == len(diagnostic_wrappers)
        == 1
    )
    assert isinstance(schedule_wrappers[0], AnimaRegionalLoraScheduleDiffusionWrapper)
    assert isinstance(phase_wrappers[0], AnimaCompositionPhaseDiffusionWrapper)
    assert isinstance(activation_wrappers[0], AnimaActivationDiffusionWrapper)
    assert isinstance(diagnostic_wrappers[0], AnimaRegionalDiagnosticsDiffusionWrapper)
    assert derived.get_all_wrappers("diffusion_model")[:4] == [
        phase_wrappers[0],
        schedule_wrappers[0],
        activation_wrappers[0],
        diagnostic_wrappers[0],
    ]
    assert derived.get_callbacks(
        CallbacksMP.ON_DETACH,
        "simple_syrup.anima_regional_lora_device_cache",
    ) == [cache_lifecycle.callback]
    assert derived.get_callbacks(
        CallbacksMP.ON_DETACH,
        "simple_syrup.anima_attention_device_cache",
    ) == [attention_cache_lifecycle.callback]
    assert derived.get_callbacks(
        CallbacksMP.ON_DETACH,
        "simple_syrup.anima_regional_lora_schedule_session",
    ) == [schedule_cache_lifecycle.callback]
    derived.patch_model(load_weights=False)
    try:
        for block_index, block_surface in enumerate(anima_surface.blocks):
            active_block = anima_surface.diffusion_model.blocks[block_index]
            assert isinstance(active_block, AnimaRegionalLoraBlockPatch)
            assert (
                active_block.layer_norm_self_attn
                is block_surface.block.layer_norm_self_attn
            )
            assert "original" not in dict(active_block.named_modules())
            assert isinstance(active_block.self_attn, AnimaRegionalSelfAttentionPatch)
            assert isinstance(active_block.cross_attn, AnimaRegionalCrossAttentionPatch)
        for target_surface in anima_surface.lora_targets:
            active_target = _resolve(
                anima_surface.diffusion_model, target_surface.target_name
            )
            assert isinstance(active_target, AnimaRegionalLoraCompositionLinearPatch)
            assert active_target.weight is active_target._backing.module.weight
            assert active_target.weight is not target_surface.module.weight
            assert "original" not in dict(active_target.named_modules())
            assert len(active_target.groups) == 4
            assert all(
                group.target.family is target_surface.family
                for group in active_target.groups
            )
    finally:
        derived.unpatch_model(unpatch_weights=False)

    for block_index, block_surface in enumerate(anima_surface.blocks):
        assert anima_surface.diffusion_model.blocks[block_index] is block_surface.block
        assert block_surface.block.self_attn is block_surface.self_attention
        assert block_surface.block.cross_attn is block_surface.cross_attention
    for target_surface in anima_surface.lora_targets:
        assert (
            _resolve(anima_surface.diffusion_model, target_surface.target_name)
            is target_surface.module
        )
    assert (
        Anima.forward,
        Block.forward,
        Attention.forward,
        GPT2FeedForward.forward,
    ) == class_forwards


def test_schedule_wrapper_resolves_before_nested_anima_execution(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Publish exact strengths only inside calls and reject nonuniform sigmas."""

    source = _patcher(_AnimaModelRoot(anima_surface.diffusion_model))
    composition = _composition(source, _admission())
    context = AnimaRegionalLoraScheduleContext()
    wrapper = AnimaRegionalLoraScheduleDiffusionWrapper(
        anima_surface,
        composition,
        context,
        _FullLoraPhaseContext(),
    )
    executor = _ScheduleRecordingExecutor(anima_surface.diffusion_model, context)
    sample_sigmas = torch.tensor([100.0, 50.0, 0.0])

    result = wrapper(
        executor,
        transformer_options={
            "sample_sigmas": sample_sigmas,
            "sigmas": torch.tensor([100.0, 100.0]),
        },
    )

    assert result == "prediction"
    assert wrapper._session.get() is not None
    assert executor.strengths == [
        tuple(
            execution.adapter_plan.model_strength
            for execution in composition.executions
        )
    ]
    with pytest.raises(RuntimeError, match="unavailable outside"):
        context.require_current()
    with pytest.raises(ValueError, match="must be uniform"):
        wrapper(
            executor,
            transformer_options={
                "sample_sigmas": sample_sigmas,
                "sigmas": torch.tensor([100.0, 50.0]),
            },
        )
    wrapper.clear(object(), True)
    assert wrapper._session.get() is None


def _admission() -> AnimaLoraAdmission:
    """Build allocation-light rank-one weights for every admitted Anima target."""

    targets: list[AnimaLoraTarget] = []
    scalar = torch.ones(())
    for block_index in range(ANIMA_BLOCK_COUNT):
        for family in AnimaLoraTargetFamily:
            input_features, output_features = expected_anima_lora_features(family)
            name = anima_lora_target_name(block_index, family)
            adapter = StandardLoraTarget(
                name,
                scalar.expand(1, input_features),
                scalar.expand(output_features, 1),
                rank=1,
                input_features=input_features,
                output_features=output_features,
            )
            targets.append(AnimaLoraTarget(block_index, family, adapter))
    return AnimaLoraAdmission(tuple(targets))


def _composition(
    model: object,
    admission: AnimaLoraAdmission,
) -> AnimaRegionalLoraComposition:
    """Build four ordered adapters sharing one attention and admitted surface."""

    context = torch.zeros((1, 1, 1024))
    contexts = BatchedRegionalAttentionContexts(
        1,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
        context,
        single_entry_regions((context.clone(),)),
    )
    mask = torch.ones((1, 1, 1))
    attention = AnimaRegionalAttentionExecution(
        contexts,
        RegionalMaskBank(mask.clone(), mask.clone(), 1, 1),
        (1.0,),
    )
    cache = RegionalLoraExecutionCache()
    return AnimaRegionalLoraComposition(
        tuple(
            AnimaRegionalLoraAdapterExecution(
                RegionalLoraAdapterPlan(
                    RegionalLoraAdapterIdentity(f"adapter-{index}.safetensors"),
                    composition_index=index,
                    region_index=0,
                    branch=RegionalLoraBranch.POSITIVE,
                    model_strength=strength,
                    schedule=(RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
                ),
                admission,
                attention,
                ModelCloneLineage.from_model(model),
                cache,
            )
            for index, strength in enumerate((0.1, 0.2, 0.3, 0.4))
        )
    )


def _resolve(diffusion_model: nn.Module, target_name: str) -> object:
    """Resolve one canonical target beneath the already-selected diffusion model."""

    segments = target_name.split(".")[1:]
    current: object = diffusion_model
    for segment in segments:
        current = getattr(current, segment)
    return current


def _patcher(model: nn.Module) -> Any:
    """Create a real CPU Comfy MODEL patcher for object-patch lifecycle proof."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)
