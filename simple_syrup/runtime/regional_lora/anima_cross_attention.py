# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluate and blend complete regional Anima cross-attention branches."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ...domain.regional_attention_batch import BatchedRegionalAttentionContexts
from ...domain.regional_attention_weights import RegionalAttentionWeightingPolicy
from ...domain.regional_conditioning_output import (
    REGIONAL_CONDITIONING_OUTPUT_COMBINER,
    RegionalConditioningOutputCombiner,
)
from ..model_patcher_mutations import ModelExactObjectPatchMutation
from .anima_activation_context import (
    ANIMA_ACTIVATION_CONTEXT,
    AnimaActivationContext,
    AnimaActivationGeometry,
)
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_branch_batch import (
    ANIMA_BASE_BRANCH_KEY,
    AnimaRegionalBranchKey,
)
from .anima_composition_phase_context import AnimaCompositionPhaseContext
from .anima_cross_attention_context import (
    AnimaCrossAttentionInvocationContext,
)
from .anima_cross_attention_weights import (
    ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY,
)
from .anima_host_module_backing import AnimaHostModuleBacking
from .anima_module_surface import AnimaModuleSurface
from .anima_query_activity import (
    ANIMA_REGIONAL_QUERY_ACTIVITY_CONTEXT,
    AnimaRegionalQueryActivityContext,
)

_CROSS_ATTENTION_CHILD_ATTRIBUTES = frozenset(
    {
        "q_proj",
        "q_norm",
        "k_proj",
        "k_norm",
        "v_proj",
        "output_proj",
        "output_dropout",
    }
)


class AnimaRegionalCrossAttentionPatch(nn.Module):
    """Batch complete attention branches and return one spatially blended output."""

    def __init__(
        self,
        original: nn.Module,
        execution: AnimaRegionalAttentionExecution,
        *,
        activation_context: AnimaActivationContext,
        invocation_context: AnimaCrossAttentionInvocationContext,
        phase_context: AnimaCompositionPhaseContext,
        query_activity: AnimaRegionalQueryActivityContext = (
            ANIMA_REGIONAL_QUERY_ACTIVITY_CONTEXT
        ),
        weighting: RegionalAttentionWeightingPolicy | None = None,
        entry_combiner: RegionalConditioningOutputCombiner | None = None,
    ) -> None:
        """Retain the exact original attention owner and focused collaborators."""

        super().__init__()
        if not isinstance(original, nn.Module):
            raise TypeError("Original Anima cross-attention must be a module.")
        if not isinstance(execution, AnimaRegionalAttentionExecution):
            raise TypeError("Anima cross-attention execution has an invalid type.")
        self._backing = AnimaHostModuleBacking(original)
        self._backing.install_children(self, _CROSS_ATTENTION_CHILD_ATTRIBUTES)
        self._execution = execution
        self._activation_context = activation_context
        self._invocation_context = invocation_context
        if not isinstance(phase_context, AnimaCompositionPhaseContext):
            raise TypeError("Anima cross-attention requires a phase context.")
        self._phase_context = phase_context
        if not isinstance(query_activity, AnimaRegionalQueryActivityContext):
            raise TypeError("Anima cross-attention requires a query-activity context.")
        self._query_activity = query_activity
        self._weighting = weighting or ANIMA_CROSS_ATTENTION_WEIGHTING_POLICY
        self._entry_combiner = entry_combiner or REGIONAL_CONDITIONING_OUTPUT_COMBINER

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep later exact child patches synchronized with installed attention."""

        backing = self.__dict__.get("_backing")
        if name in _CROSS_ATTENTION_CHILD_ATTRIBUTES and isinstance(
            backing, AnimaHostModuleBacking
        ):
            backing.replace_public_attribute(self, name, value)
            return
        super().__setattr__(name, value)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Evaluate all complete branches once and blend before the residual path."""

        geometry = self._activation_context.require_current()
        execution_contexts = self._execution.active_contexts
        self._validate_inputs(
            x=x,
            context=context,
            geometry=geometry,
            execution_contexts=execution_contexts,
        )
        if context is None:
            raise AssertionError("Validated cross-attention context became unavailable")
        activity = self._query_activity.resolve(
            self._execution,
            geometry,
            device=x.device,
            dtype=x.dtype,
        )
        weights = activity.attention_weights
        branch_values: dict[AnimaRegionalBranchKey, torch.Tensor] = {
            ANIMA_BASE_BRANCH_KEY: context
        }
        for region in execution_contexts.regions:
            for entry in region.entries:
                key = AnimaRegionalBranchKey(
                    region.region_index,
                    entry.entry_index,
                )
                branch_values[key] = entry.context
        branch_batch = activity.attention_branches
        branch_x = branch_batch.pack_source(x)
        branch_context = branch_batch.pack_branch_values(branch_values)
        with self._invocation_context.activate(branch_batch.invocation):
            branch_output = self._backing.module(
                branch_x,
                branch_context,
                rope_emb=rope_emb,
                transformer_options=(
                    {} if transformer_options is None else transformer_options
                ),
            )
        if not isinstance(branch_output, torch.Tensor):
            raise TypeError("Original Anima cross-attention must return a tensor.")
        expected_shape = (branch_batch.packed_batch_size, *x.shape[1:])
        if tuple(branch_output.shape) != expected_shape:
            raise ValueError(
                "Original Anima cross-attention returned an invalid branch batch; "
                f"expected {expected_shape}, observed {tuple(branch_output.shape)}."
            )
        restored = branch_batch.restore(branch_output)
        empty_output = torch.zeros_like(x)
        base_output = restored.get(ANIMA_BASE_BRANCH_KEY, empty_output)
        regional_outputs: list[torch.Tensor] = []
        for region in execution_contexts.regions:
            regional_outputs.append(
                self._entry_combiner.combine(
                    tuple(
                        restored.get(
                            AnimaRegionalBranchKey(
                                region.region_index,
                                entry.entry_index,
                            ),
                            empty_output,
                        )
                        for entry in region.entries
                    ),
                    strengths=tuple(entry.strengths for entry in region.entries),
                )
            )
        return self._weighting.blend(
            weights=weights,
            base_output=base_output,
            regional_outputs=torch.stack(regional_outputs),
        )

    def _validate_inputs(
        self,
        *,
        x: torch.Tensor,
        context: torch.Tensor | None,
        geometry: AnimaActivationGeometry,
        execution_contexts: BatchedRegionalAttentionContexts,
    ) -> None:
        """Require exact query, batch, and context alignment before attention work."""

        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("Anima cross-attention query must use BxQxD layout.")
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError("Anima cross-attention context must use BxSxD layout.")
        aligned = execution_contexts
        expected_batch = int(aligned.base_context.shape[0])
        if int(x.shape[0]) != expected_batch:
            raise ValueError(
                "Anima cross-attention query batch does not match aligned contexts."
            )
        if int(x.shape[1]) != geometry.query_token_count:
            raise ValueError(
                "Anima cross-attention query length does not match activation geometry."
            )
        if geometry.input_batch_size != expected_batch:
            raise ValueError(
                "Anima activation batch does not match aligned attention contexts."
            )
        if (
            context.shape != aligned.base_context.shape
            or context.dtype != aligned.base_context.dtype
            or context.device != aligned.base_context.device
        ):
            raise ValueError(
                "Anima base cross-attention context does not match aligned base state."
            )
        for region in aligned.regions:
            for entry in region.entries:
                if (
                    entry.context.shape != context.shape
                    or entry.context.dtype != context.dtype
                    or entry.context.device != context.device
                ):
                    raise ValueError(
                        "Anima regional cross-attention context "
                        f"{region.region_index} entry {entry.entry_index} does not "
                        "match the active base context."
                    )


def anima_cross_attention_mutations(
    surface: AnimaModuleSurface,
    execution: AnimaRegionalAttentionExecution,
    *,
    activation_context: AnimaActivationContext = ANIMA_ACTIVATION_CONTEXT,
    invocation_context: AnimaCrossAttentionInvocationContext,
    phase_context: AnimaCompositionPhaseContext,
    query_activity: AnimaRegionalQueryActivityContext = (
        ANIMA_REGIONAL_QUERY_ACTIVITY_CONTEXT
    ),
) -> tuple[ModelExactObjectPatchMutation, ...]:
    """Build one exact clone-local cross-attention replacement per Anima block."""

    if not isinstance(surface, AnimaModuleSurface):
        raise TypeError("Anima cross-attention mutations require a module surface.")
    return tuple(
        ModelExactObjectPatchMutation(
            path=f"diffusion_model.blocks.{block.block_index}.cross_attn",
            expected_object=block.cross_attention,
            replacement=AnimaRegionalCrossAttentionPatch(
                block.cross_attention,
                execution,
                activation_context=activation_context,
                invocation_context=invocation_context,
                phase_context=phase_context,
                query_activity=query_activity,
            ),
        )
        for block in surface.blocks
    )
