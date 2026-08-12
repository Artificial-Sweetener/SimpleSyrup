# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluate and spatially blend complete Anima AdaLN regional branches."""

from __future__ import annotations

import torch
from torch import nn

from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_branch_batch import (
    ANIMA_BASE_BRANCH_KEY,
    AnimaRegionalBranchKey,
)
from .anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
)
from .anima_query_activity import AnimaRegionalQueryActivity

_ModulationTriplet = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class AnimaRegionalAdalnEvaluator:
    """Own exact base/region AdaLN evaluation and complete-output blending."""

    def __init__(
        self,
        attention: AnimaRegionalAttentionExecution,
        branch_context: AnimaLoraBranchInvocationContext,
    ) -> None:
        """Retain the adapter execution and task-local branch authority."""

        self._attention = attention
        self._branch_context = branch_context

    def evaluate_all(
        self,
        modules: tuple[nn.Sequential, nn.Sequential, nn.Sequential],
        embedding: torch.Tensor,
        built_in_lora: torch.Tensor,
        activity: AnimaRegionalQueryActivity,
    ) -> tuple[_ModulationTriplet, _ModulationTriplet, _ModulationTriplet]:
        """Evaluate three stacks from one shared stateless activation output."""

        branch_batch = activity.adaln_branches
        if branch_batch.source_batch_size != int(embedding.shape[0]):
            raise ValueError("Anima AdaLN branch plan does not match embedding batch.")
        with self._branch_context.activate(branch_batch.invocation):
            activated = modules[0][0](branch_batch.pack_source(embedding))
            branch_outputs = tuple(
                module[2](module[1](activated)) for module in modules
            )
        expected_shape = (
            branch_batch.packed_batch_size,
            int(embedding.shape[1]),
            int(embedding.shape[-1]) * 3,
        )
        if any(
            not isinstance(output, torch.Tensor)
            or tuple(output.shape) != expected_shape
            for output in branch_outputs
        ):
            raise ValueError(
                "Anima AdaLN branch output has an invalid compact batch shape."
            )
        blended = tuple(
            self._blend(
                branch_batch.restore(branch_output),
                embedding=embedding,
                built_in_lora=built_in_lora,
                activity=activity,
            )
            for branch_output in branch_outputs
        )
        return blended[0], blended[1], blended[2]

    @staticmethod
    def _blend(
        branch_outputs: dict[AnimaRegionalBranchKey, torch.Tensor],
        *,
        embedding: torch.Tensor,
        built_in_lora: torch.Tensor,
        activity: AnimaRegionalQueryActivity,
    ) -> _ModulationTriplet:
        """Preserve built-in LoRA and spatially blend one branch output."""

        batch = int(embedding.shape[0])
        expected_features = int(embedding.shape[-1]) * 3
        if tuple(built_in_lora.shape) != (
            batch,
            int(embedding.shape[1]),
            expected_features,
        ):
            raise ValueError("Anima built-in AdaLN-LoRA shape is invalid.")
        empty_output = built_in_lora.new_zeros(built_in_lora.shape)
        base_output = (
            branch_outputs.get(ANIMA_BASE_BRANCH_KEY, empty_output) + built_in_lora
        )
        region_outputs = torch.stack(
            tuple(
                branch_outputs.get(
                    AnimaRegionalBranchKey(region_index, 0),
                    empty_output,
                )
                + built_in_lora
                for region_index in range(int(activity.masks.masks.shape[0]))
            )
        )
        base_grid = base_output[:, :, None, None, :]
        differences = region_outputs[:, :, :, None, None, :] - base_grid.unsqueeze(0)
        typed_masks = activity.masks.masks.unsqueeze(-1).to(base_output.dtype)
        region_count = int(activity.masks.masks.shape[0])
        if region_count == 1:
            modulation = torch.addcmul(
                base_grid,
                typed_masks[0],
                differences[0],
            )
        else:
            modulation = base_grid + (typed_masks * differences).sum(dim=0)
        feature_size = int(embedding.shape[-1])
        return (
            modulation[..., :feature_size],
            modulation[..., feature_size : 2 * feature_size],
            modulation[..., 2 * feature_size :],
        )
