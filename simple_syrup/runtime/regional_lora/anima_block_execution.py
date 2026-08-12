# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run one Anima block with spatial regional LoRA and exact AdaLN branches."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .anima_activation_context import (
    AnimaActivationContext,
    AnimaActivationGeometry,
)
from .anima_adaln import AnimaRegionalAdalnEvaluator
from .anima_attention_execution import AnimaRegionalAttentionExecution
from .anima_execution_scope import (
    AnimaLoraBranchInvocationContext,
    AnimaLoraSpatialInvocation,
    AnimaLoraSpatialInvocationContext,
)
from .anima_host_module_backing import AnimaHostModuleBacking
from .anima_query_activity import (
    ANIMA_REGIONAL_QUERY_ACTIVITY_CONTEXT,
    AnimaRegionalQueryActivity,
    AnimaRegionalQueryActivityContext,
)
from .anima_schedule_context import AnimaRegionalLoraScheduleContext

_BLOCK_CHILD_ATTRIBUTES = frozenset(
    {
        "layer_norm_self_attn",
        "self_attn",
        "layer_norm_cross_attn",
        "cross_attn",
        "layer_norm_mlp",
        "mlp",
        "adaln_modulation_self_attn",
        "adaln_modulation_cross_attn",
        "adaln_modulation_mlp",
    }
)


class AnimaRegionalLoraBlockPatch(nn.Module):
    """Preserve one block trajectory while spatializing one adapter's deltas."""

    def __init__(
        self,
        original: nn.Module,
        attention: AnimaRegionalAttentionExecution,
        *,
        activation_context: AnimaActivationContext,
        branch_context: AnimaLoraBranchInvocationContext,
        spatial_context: AnimaLoraSpatialInvocationContext,
        schedule_context: AnimaRegionalLoraScheduleContext,
        query_activity: AnimaRegionalQueryActivityContext = (
            ANIMA_REGIONAL_QUERY_ACTIVITY_CONTEXT
        ),
    ) -> None:
        """Retain the exact block and focused regional execution collaborators."""

        super().__init__()
        if not isinstance(original, nn.Module):
            raise TypeError("Original Anima block must be a module.")
        self._backing = AnimaHostModuleBacking(original)
        self._backing.install_children(self, _BLOCK_CHILD_ATTRIBUTES)
        self._attention = attention
        self._activation_context = activation_context
        self._spatial_context = spatial_context
        if not isinstance(schedule_context, AnimaRegionalLoraScheduleContext):
            raise TypeError("Anima regional LoRA block requires a schedule context.")
        self._schedule_context = schedule_context
        if not isinstance(query_activity, AnimaRegionalQueryActivityContext):
            raise TypeError("Anima regional LoRA block requires query activity.")
        self._query_activity = query_activity
        self._adaln = AnimaRegionalAdalnEvaluator(attention, branch_context)

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep later exact child patches synchronized with the retained block."""

        backing = self.__dict__.get("_backing")
        if name in _BLOCK_CHILD_ATTRIBUTES and isinstance(
            backing, AnimaHostModuleBacking
        ):
            backing.replace_public_attribute(self, name, value)
            return
        super().__setattr__(name, value)

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: torch.Tensor | None = None,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
        extra_per_block_pos_emb: torch.Tensor | None = None,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Execute installed block stages once with spatial regional modulation."""

        if not self._schedule_context.require_current().has_active_strength:
            output = self._backing.module(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_per_block_pos_emb,
                transformer_options=transformer_options,
            )
            if not isinstance(output, torch.Tensor):
                raise TypeError("Retained inactive Anima block returned a non-tensor.")
            return output
        options = {} if transformer_options is None else transformer_options
        geometry = self._activation_context.require_current()
        self._validate_inputs(
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            adaln_lora_B_T_3D,
            geometry,
        )
        if not isinstance(adaln_lora_B_T_3D, torch.Tensor):
            raise AssertionError("Validated built-in AdaLN-LoRA became unavailable")
        activity = self._query_activity.resolve(
            self._attention,
            geometry,
            device=x_B_T_H_W_D.device,
            dtype=x_B_T_H_W_D.dtype,
        )
        if extra_per_block_pos_emb is not None:
            if extra_per_block_pos_emb.shape != x_B_T_H_W_D.shape:
                raise ValueError("Anima per-block position embedding shape changed.")
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        with self._spatial_context.activate(AnimaLoraSpatialInvocation(activity.masks)):
            return self._execute_stages(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                transformer_options=options,
                activity=activity,
            )

    def _execute_stages(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        cross_attention_context: torch.Tensor,
        *,
        rope_emb_L_1_1_D: torch.Tensor | None,
        adaln_lora_B_T_3D: torch.Tensor,
        transformer_options: dict[str, Any],
        activity: AnimaRegionalQueryActivity,
    ) -> torch.Tensor:
        """Apply self-attention, cross-attention, and MLP in installed order."""

        residual_dtype = x.dtype
        compute_dtype = embedding.dtype
        (
            (self_shift, self_scale, self_gate),
            (cross_shift, cross_scale, cross_gate),
            (mlp_shift, mlp_scale, mlp_gate),
        ) = self._adaln.evaluate_all(
            (
                self._adaln_module("adaln_modulation_self_attn"),
                self._adaln_module("adaln_modulation_cross_attn"),
                self._adaln_module("adaln_modulation_mlp"),
            ),
            embedding,
            adaln_lora_B_T_3D,
            activity,
        )

        normalized = self._modulated_norm(
            x,
            self._module("layer_norm_self_attn"),
            self_scale,
            self_shift,
        )
        batch, time, height, width, features = normalized.shape
        self_result = self._module("self_attn")(
            normalized.to(compute_dtype).reshape(
                batch, time * height * width, features
            ),
            None,
            rope_emb=rope_emb_L_1_1_D,
            transformer_options=transformer_options,
        )
        self_result = self._attention_grid(
            self_result,
            batch=batch,
            time=time,
            height=height,
            width=width,
            features=features,
            name="self-attention",
        )
        x = torch.addcmul(
            x,
            self_gate.to(residual_dtype),
            self_result.to(residual_dtype),
        )

        normalized = self._modulated_norm(
            x,
            self._module("layer_norm_cross_attn"),
            cross_scale,
            cross_shift,
        )
        cross_result = self._module("cross_attn")(
            normalized.to(compute_dtype).reshape(
                batch,
                time * height * width,
                features,
            ),
            cross_attention_context,
            rope_emb=rope_emb_L_1_1_D,
            transformer_options=transformer_options,
        )
        cross_result = self._attention_grid(
            cross_result,
            batch=batch,
            time=time,
            height=height,
            width=width,
            features=features,
            name="cross-attention",
        )
        x = torch.addcmul(
            x,
            cross_gate.to(residual_dtype),
            cross_result.to(residual_dtype),
        )

        normalized = self._modulated_norm(
            x,
            self._module("layer_norm_mlp"),
            mlp_scale,
            mlp_shift,
        ).to(compute_dtype)
        normalized = self._apply_mlp_input_patches(normalized, transformer_options)
        mlp_result = self._module("mlp")(normalized)
        if not isinstance(mlp_result, torch.Tensor) or mlp_result.shape != x.shape:
            raise ValueError("Anima MLP output shape does not match the residual.")
        return torch.addcmul(
            x,
            mlp_gate.to(residual_dtype),
            mlp_result.to(residual_dtype),
        )

    def _module(self, name: str) -> nn.Module:
        """Return one required retained block child through a narrowed type."""

        module = getattr(self, name, None)
        if not isinstance(module, nn.Module):
            raise TypeError(f"Anima block child {name} must be a module.")
        return module

    def _adaln_module(self, name: str) -> nn.Sequential:
        """Return one validated three-stage AdaLN modulation owner."""

        module = self._module(name)
        if not isinstance(module, nn.Sequential) or len(module) != 3:
            raise TypeError(f"Anima block child {name} must be a three-stage sequence.")
        return module

    @staticmethod
    def _modulated_norm(
        x: torch.Tensor,
        normalization: nn.Module,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one spatial AdaLN scale and shift to the shared residual."""

        normalized = normalization(x)
        if not isinstance(normalized, torch.Tensor) or normalized.shape != x.shape:
            raise ValueError("Anima block normalization returned an invalid shape.")
        return normalized * (1 + scale) + shift

    @staticmethod
    def _attention_grid(
        output: object,
        *,
        batch: int,
        time: int,
        height: int,
        width: int,
        features: int,
        name: str,
    ) -> torch.Tensor:
        """Validate and restore one flattened attention result to B/T/H/W/D."""

        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (
            batch,
            time * height * width,
            features,
        ):
            raise ValueError(f"Anima {name} output has an invalid shape.")
        return output.reshape(batch, time, height, width, features)

    @staticmethod
    def _apply_mlp_input_patches(
        inputs: torch.Tensor,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Preserve installed Comfy MLP input-patch behavior and ordering."""

        patches = transformer_options.get("patches", {})
        if not isinstance(patches, dict):
            raise TypeError("Anima transformer patches must be a dictionary.")
        mlp_patches = patches.get("mlp_patch", ())
        if not isinstance(mlp_patches, list | tuple) or any(
            not callable(patch) for patch in mlp_patches
        ):
            raise TypeError("Anima MLP patches must be callable values.")
        patched = inputs
        for patch in mlp_patches:
            result = patch({"x": patched, "transformer_options": transformer_options})
            if not isinstance(result, dict) or not isinstance(
                result.get("x"), torch.Tensor
            ):
                raise TypeError("Anima MLP patch must return a tensor under 'x'.")
            patched = result["x"]
        return patched

    def _validate_inputs(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        cross_attention_context: torch.Tensor,
        built_in_lora: torch.Tensor | None,
        geometry: AnimaActivationGeometry,
    ) -> None:
        """Require exact installed block tensor and activation alignment."""

        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError("Anima block residual must use BxTxHxWxD layout.")
        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 3:
            raise ValueError("Anima block embedding must use BxTxD layout.")
        if not isinstance(cross_attention_context, torch.Tensor):
            raise TypeError("Anima block cross-attention context must be a tensor.")
        if not isinstance(built_in_lora, torch.Tensor):
            raise TypeError("Anima block requires its built-in AdaLN-LoRA tensor.")
        if tuple(x.shape[:4]) != (
            geometry.input_batch_size,
            geometry.query_time,
            geometry.query_height,
            geometry.query_width,
        ):
            raise ValueError("Anima block residual does not match activation geometry.")
        if tuple(embedding.shape[:2]) != tuple(x.shape[:2]):
            raise ValueError(
                "Anima block embedding batch/time does not match residual."
            )
        if int(embedding.shape[-1]) != int(x.shape[-1]):
            raise ValueError("Anima block embedding features do not match residual.")
