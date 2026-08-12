# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assemble exact regional-LoRA linear mutations for installed Anima graphs."""

from __future__ import annotations

from ..model_patcher_mutations import ModelExactObjectPatchMutation
from ..patcher_lifecycle import ModelMutation
from .anima_composition import AnimaRegionalLoraComposition
from .anima_device_cache_lifecycle import AnimaRegionalLoraDeviceCacheLifecycle
from .anima_linear_execution import AnimaRegionalLoraCompositionLinearPatch
from .anima_lora_weights import AnimaLoraWeightResolver
from .anima_module_surface import AnimaModuleSurface
from .anima_projection_batch import AnimaProjectionBatchRegistry
from .anima_schedule_context import AnimaRegionalLoraScheduleContext


def anima_lora_composition_linear_mutations(
    surface: AnimaModuleSurface,
    composition: AnimaRegionalLoraComposition,
    *,
    weight_resolver: AnimaLoraWeightResolver,
    schedule_context: AnimaRegionalLoraScheduleContext,
) -> tuple[ModelMutation, ...]:
    """Build one exact multi-adapter replacement per active Anima target."""

    surface_targets = {
        descriptor.target_name: descriptor for descriptor in surface.lora_targets
    }
    for execution in composition.executions:
        for target in execution.admission.targets:
            descriptor = surface_targets.get(target.adapter.target)
            if descriptor is None:
                raise ValueError(
                    "Anima LoRA composition target-key mismatch: admitted target "
                    f"{target.adapter.target!r} is absent from the verified surface."
                )
            if (
                target.block_index != descriptor.block_index
                or target.family is not descriptor.family
            ):
                raise ValueError(
                    "Anima LoRA composition target descriptor mismatch for "
                    f"{target.adapter.target!r}: expected block "
                    f"{descriptor.block_index} family {descriptor.family.value}, "
                    f"observed block {target.block_index} family "
                    f"{target.family.value}."
                )
    registry = AnimaProjectionBatchRegistry()
    projection_groups = _projection_groups(surface)
    projection_target_names = frozenset(
        name for _group_id, names in projection_groups for name in names
    )
    patches: dict[str, AnimaRegionalLoraCompositionLinearPatch] = {}
    mutations: list[ModelMutation] = []
    for descriptor in surface.lora_targets:
        groups = composition.groups_for_target(descriptor.target_name)
        if not groups:
            continue
        patch = AnimaRegionalLoraCompositionLinearPatch(
            descriptor.module,
            groups,
            weight_resolver=weight_resolver,
            schedule_context=schedule_context,
            projection_batches=(
                registry if descriptor.target_name in projection_target_names else None
            ),
        )
        patches[descriptor.target_name] = patch
        mutations.append(
            ModelExactObjectPatchMutation(
                path=descriptor.target_name,
                expected_object=descriptor.module,
                replacement=patch,
            )
        )
    for group_id, names in projection_groups:
        participants = tuple(patches[name] for name in names if name in patches)
        if len(participants) == len(names):
            registry.register(group_id, participants)
    lifecycle = AnimaRegionalLoraDeviceCacheLifecycle(
        tuple(patches.values()),
        composition,
        weight_resolver,
        registry,
    )
    mutations.append(lifecycle.mutation())
    return tuple(mutations)


def _projection_groups(
    surface: AnimaModuleSurface,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Describe exact shared-input Anima projection groups once per assembly."""

    groups: list[tuple[str, tuple[str, ...]]] = []
    for block in surface.blocks:
        self_names = tuple(
            f"diffusion_model.blocks.{block.block_index}.self_attn.{name}_proj"
            for name in ("q", "k", "v")
        )
        cross_names = tuple(
            f"diffusion_model.blocks.{block.block_index}.cross_attn.{name}_proj"
            for name in ("k", "v")
        )
        adaln_names = tuple(
            f"diffusion_model.blocks.{block.block_index}.adaln_modulation_{stage}.1"
            for stage in ("self_attn", "cross_attn", "mlp")
        )
        groups.extend(
            (
                (f"block_{block.block_index}_self_qkv", self_names),
                (f"block_{block.block_index}_cross_kv", cross_names),
                (f"block_{block.block_index}_adaln_first", adaln_names),
            )
        )
    return tuple(groups)
