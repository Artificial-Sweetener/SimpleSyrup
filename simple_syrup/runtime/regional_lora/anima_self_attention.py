# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply scheduled image-token ownership to installed Anima self-attention."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from ..model_patcher_mutations import ModelExactObjectPatchMutation
from .anima_activation_context import AnimaActivationContext
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_composition_phase_context import AnimaCompositionPhaseContext
from .anima_host_module_backing import AnimaHostModuleBacking
from .anima_module_surface import AnimaModuleSurface
from .anima_query_activity import AnimaRegionalQueryActivityContext
from .anima_self_attention_partition_cache import AnimaSelfAttentionPartitionCache
from .anima_self_attention_partition_execution import (
    ANIMA_SELF_ATTENTION_PARTITION_EXECUTION,
    AnimaSelfAttentionPartitionExecution,
)

_SELF_ATTENTION_CHILD_ATTRIBUTES = frozenset(
    {
        "q_proj",
        "q_norm",
        "k_proj",
        "k_norm",
        "v_proj",
        "v_norm",
        "output_proj",
        "output_dropout",
    }
)


class AnimaRegionalSelfAttentionPatch(nn.Module):
    """Restrict early image attention and preserve exact late host execution."""

    def __init__(
        self,
        original: nn.Module,
        execution: AnimaRegionalAttentionExecution,
        *,
        activation_context: AnimaActivationContext,
        phase_context: AnimaCompositionPhaseContext,
        query_activity: AnimaRegionalQueryActivityContext,
        partition_cache: AnimaSelfAttentionPartitionCache,
        partition_execution: AnimaSelfAttentionPartitionExecution = (
            ANIMA_SELF_ATTENTION_PARTITION_EXECUTION
        ),
    ) -> None:
        """Retain installed attention and focused ownership collaborators."""

        super().__init__()
        if not isinstance(original, nn.Module):
            raise TypeError("Original Anima self-attention must be a module.")
        self._backing = AnimaHostModuleBacking(original)
        self._backing.install_children(self, _SELF_ATTENTION_CHILD_ATTRIBUTES)
        self._execution = execution
        self._activation_context = activation_context
        self._phase_context = phase_context
        self._query_activity = query_activity
        if not isinstance(partition_cache, AnimaSelfAttentionPartitionCache):
            raise TypeError("Anima self-attention requires a partition cache.")
        if not isinstance(partition_execution, AnimaSelfAttentionPartitionExecution):
            raise TypeError("Anima self-attention requires partition execution.")
        self._partition_cache = partition_cache
        self._partition_execution = partition_execution

    def __setattr__(self, name: str, value: Any) -> None:
        """Synchronize later exact child patches with installed attention."""

        backing = self.__dict__.get("_backing")
        if name in _SELF_ATTENTION_CHILD_ATTRIBUTES and isinstance(
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
        """Execute exact global or additive-bias self-attention by schedule state."""

        if context is not None:
            raise ValueError("Anima self-attention context must be None.")
        options = {} if transformer_options is None else transformer_options
        if not self._phase_context.require_current().restrict_self_attention:
            output = self._backing.module(
                x,
                None,
                rope_emb=rope_emb,
                transformer_options=options,
            )
            if not isinstance(output, torch.Tensor):
                raise TypeError("Installed Anima self-attention returned a non-tensor.")
            return output
        geometry = self._activation_context.require_current()
        if x.ndim != 3 or tuple(x.shape[:2]) != (
            geometry.input_batch_size,
            geometry.query_token_count,
        ):
            raise ValueError(
                "Anima self-attention query must match active B/Q geometry."
            )
        activity = self._query_activity.resolve(
            self._execution,
            geometry,
            device=x.device,
            dtype=x.dtype,
        )
        plan = self._partition_cache.resolve(
            activity.masks.flattened,
            height=geometry.query_height,
            width=geometry.query_width,
        )
        host = cast(Any, self._backing.module)
        q, k, v = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            host.compute_qkv(
                x,
                None,
                rope_emb=rope_emb,
                transformer_options=options,
            ),
        )
        attended = self._partition_execution.execute(
            q,
            k,
            v,
            plan,
            attention=lambda packed_q, packed_k, packed_v: cast(
                torch.Tensor,
                host.attn_op(
                    packed_q,
                    packed_k,
                    packed_v,
                    transformer_options=options,
                ),
            ),
        )
        output_projection = cast(nn.Module, self.output_proj)
        output_dropout = cast(nn.Module, self.output_dropout)
        output = output_dropout(output_projection(attended))
        if not isinstance(output, torch.Tensor):
            raise TypeError("Masked Anima self-attention returned a non-tensor.")
        if output.shape != x.shape:
            raise ValueError("Masked Anima self-attention returned an invalid shape.")
        return output


def anima_self_attention_mutations(
    surface: AnimaModuleSurface,
    execution: AnimaRegionalAttentionExecution,
    *,
    activation_context: AnimaActivationContext,
    phase_context: AnimaCompositionPhaseContext,
    query_activity: AnimaRegionalQueryActivityContext,
    partition_cache: AnimaSelfAttentionPartitionCache,
) -> tuple[ModelExactObjectPatchMutation, ...]:
    """Build one exact clone-local self-attention replacement per Anima block."""

    return tuple(
        ModelExactObjectPatchMutation(
            path=f"diffusion_model.blocks.{block.block_index}.self_attn",
            expected_object=block.self_attention,
            replacement=AnimaRegionalSelfAttentionPatch(
                block.self_attention,
                execution,
                activation_context=activation_context,
                phase_context=phase_context,
                query_activity=query_activity,
                partition_cache=partition_cache,
            ),
        )
        for block in surface.blocks
    )
