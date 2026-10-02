# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose shared inversion controls on maintained implementation-backed samplers."""

from __future__ import annotations

from typing import Any

from ..nodes.detailer_input_adapters import (
    float_input,
    int_input,
    str_input,
)
from .legacy_node_adapter import LegacyNodeV3Adapter
from .sampler_options_schema import (
    COMFY_IO,
    inversion_from_controls,
    noise_inversion_inputs,
)


class LegacyInversionNodeV3Adapter(LegacyNodeV3Adapter):
    """Normalize direct or list-mode widgets into the shared inversion domain value."""

    @classmethod
    def define_schema(cls) -> Any:
        """Append the shared five inversion controls after the sampler inputs."""
        schema = super().define_schema()
        schema.inputs.extend(noise_inversion_inputs(COMFY_IO, convenience=True))
        return schema

    @classmethod
    def execute(cls, **kwargs: object) -> Any:
        """Narrow inversion widgets before delegating normal implementation inputs."""
        values = dict(kwargs)
        list_mode = bool(getattr(cls.LEGACY_NODE_CLASS, "INPUT_IS_LIST", False))
        operation = cls.DISPLAY_NAME
        inversion = inversion_from_controls(
            inversion_method=str_input(
                values.pop("inversion_method", "euler"),
                "inversion_method",
                list_mode,
                operation,
            ),
            inversion_resolution_scale=float_input(
                values.pop("inversion_resolution_scale", 0.5),
                "inversion_resolution_scale",
                list_mode,
                operation,
            ),
            inversion_steps=int_input(
                values.pop("inversion_steps", 2),
                "inversion_steps",
                list_mode,
                operation,
            ),
            inversion_switch_fraction=float_input(
                values.pop("inversion_switch_fraction", 0.75),
                "inversion_switch_fraction",
                list_mode,
                operation,
            ),
            inversion_finishing_steps=int_input(
                values.pop("inversion_finishing_steps", 1),
                "inversion_finishing_steps",
                list_mode,
                operation,
            ),
        )
        values["noise_inversion"] = inversion
        return super().execute(**values)
