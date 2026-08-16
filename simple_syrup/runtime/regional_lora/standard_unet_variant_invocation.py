# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Normalize Comfy standard-UNet calls for persistent regional variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch

_POSITIONAL_NAMES = (
    "input",
    "timesteps",
    "context",
    "y",
    "control",
    "transformer_options",
)
_OPTIONAL_DEFAULTS: tuple[object, ...] = (None, None, None, None)


@dataclass(frozen=True, slots=True)
class StandardUnetVariantInvocation:
    """Own one normalized Comfy diffusion invocation and graph-local copies."""

    arguments: tuple[object, ...]
    keyword_arguments: dict[str, object]
    model_input: torch.Tensor
    transformer_options: dict[str, object]

    @classmethod
    def bind(
        cls,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> StandardUnetVariantInvocation:
        """Bind keyword or positional host inputs to the UNet wrapper contract."""

        if not args or len(args) > len(_POSITIONAL_NAMES):
            raise TypeError(
                "Standard UNet variant execution requires one to six positional "
                "diffusion arguments."
            )
        model_input = args[0]
        if not isinstance(model_input, torch.Tensor) or model_input.ndim != 4:
            raise TypeError("Standard UNet variant execution requires BCHW input.")

        remaining = kwargs.copy()
        normalized = list(args)
        for index in range(len(args)):
            name = _POSITIONAL_NAMES[index]
            if name in remaining:
                raise TypeError(f"Standard UNet variant invocation repeats {name!r}.")
        for index in range(len(args), len(_POSITIONAL_NAMES)):
            name = _POSITIONAL_NAMES[index]
            if name in remaining:
                normalized.append(remaining.pop(name))
                continue
            default_index = index - 2
            if default_index < 0:
                raise TypeError("Standard UNet variant execution requires timesteps.")
            normalized.append(_OPTIONAL_DEFAULTS[default_index])

        options = normalized[5]
        if not isinstance(options, dict):
            raise TypeError(
                "Standard UNet variant execution requires transformer options."
            )
        return cls(
            tuple(normalized),
            remaining,
            model_input,
            options,
        )

    def graph_call(
        self,
        *,
        context: torch.Tensor,
        transformer_options: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        """Return arguments with exact context and graph-local mutable metadata."""

        if context.ndim != 3:
            raise ValueError("Standard UNet variant context must use BxSxD layout.")
        source_context = self.arguments[2]
        if not isinstance(source_context, torch.Tensor):
            raise TypeError("Standard UNet source context must be a tensor.")
        if (
            int(context.shape[0]) != int(self.model_input.shape[0])
            or context.shape[1:] != source_context.shape[1:]
            or context.device != source_context.device
            or context.dtype != source_context.dtype
        ):
            raise ValueError(
                "Standard UNet variant context must match the active model batch."
            )
        if not isinstance(transformer_options, dict):
            raise TypeError(
                "Standard UNet graph transformer options must be a dictionary."
            )
        arguments = list(self.arguments)
        arguments[2] = context
        arguments[4] = self._copy_control(arguments[4])
        arguments[5] = transformer_options.copy()
        return tuple(arguments), self.keyword_arguments.copy()

    @staticmethod
    def _copy_control(control: object) -> object:
        """Copy the control lists consumed destructively by standard UNets."""

        if control is None:
            return None
        if not isinstance(control, dict):
            raise TypeError(
                "Standard UNet variant execution requires mapping control data."
            )
        return {
            key: value.copy() if isinstance(value, list) else value
            for key, value in control.items()
        }
