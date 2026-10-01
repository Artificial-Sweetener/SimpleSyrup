# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare bypass-compatible sampler options sockets and inversion controls."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, cast

from ..domain.noise_inversion import (
    INVERSION_METHODS,
    InversionMethod,
    NoiseInversionOptions,
)

if TYPE_CHECKING:

    class OptionsNodeBase:
        """Describe Comfy's host-facing node metadata for strict type checking."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    OptionsNodeBase = import_module("comfy_api.latest").io.ComfyNode

COMFY_IO: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io
OPTIONS_TYPE = "SIMPLE_SYRUP_SAMPLER_OPTIONS"


def options_input(comfy_io: Any) -> Any:
    """Allow any capability to start a chain or consume a preceding capability."""
    return comfy_io.Custom(OPTIONS_TYPE).Input(
        "options",
        optional=True,
        tooltip=(
            "Optional preceding sampler options; bypass this node "
            "to omit its contribution."
        ),
    )


def options_output(comfy_io: Any) -> Any:
    """Match the input type so Comfy can bypass capability nodes natively."""
    return comfy_io.Custom(OPTIONS_TYPE).Output(
        "options",
        tooltip="Combined sampler options; connect another options node or KSampler.",
    )


def noise_inversion_inputs(comfy_io: Any, *, convenience: bool = False) -> list[Any]:
    """Use the accepted two coarse plus one finishing-step recipe in both APIs."""
    enabled = (
        [
            comfy_io.Boolean.Input(
                "noise_inversion_enabled",
                default=False,
                optional=True,
                tooltip=(
                    "Derives starting noise from the input image; "
                    "adds inversion work before denoising."
                ),
            )
        ]
        if convenience
        else []
    )
    return [
        *enabled,
        comfy_io.Combo.Input(
            "inversion_method",
            options=list(INVERSION_METHODS),
            default="euler",
            optional=convenience,
            tooltip=(
                "Euler uses one model evaluation per inversion step; "
                "Heun uses two for greater accuracy."
            ),
        ),
        comfy_io.Float.Input(
            "inversion_resolution_scale",
            default=0.5,
            min=0.01,
            max=1.0,
            step=0.05,
            optional=convenience,
            tooltip=(
                "Scales inversion width and height; "
                "0.5 uses half-sized dimensions for lower cost."
            ),
        ),
        comfy_io.Int.Input(
            "inversion_steps",
            default=2,
            min=1,
            max=64,
            optional=convenience,
            tooltip=(
                "Steps at the selected inversion resolution; "
                "more steps cost more model evaluations."
            ),
        ),
        comfy_io.Float.Input(
            "inversion_switch_fraction",
            default=0.75,
            min=0.01,
            max=1.0,
            step=0.05,
            optional=convenience,
            tooltip=(
                "Noise-level fraction reached before the full-resolution finish; "
                "0.75 means 75%."
            ),
        ),
        comfy_io.Int.Input(
            "inversion_finishing_steps",
            default=1,
            min=0,
            max=64,
            optional=convenience,
            tooltip=(
                "Full-resolution inversion steps after a reduced stage; "
                "0 finishes entirely at reduced size."
            ),
        ),
        comfy_io.Combo.Input(
            "inversion_finishing_method",
            options=list(INVERSION_METHODS),
            default="euler",
            optional=convenience,
            tooltip=(
                "Method for full-resolution finishing; "
                "Euler costs one evaluation per step, Heun two."
            ),
        ),
    ]


def inversion_from_controls(
    *,
    noise_inversion_enabled: bool,
    inversion_method: str = "euler",
    inversion_resolution_scale: float = 0.5,
    inversion_steps: int = 2,
    inversion_switch_fraction: float = 0.75,
    inversion_finishing_steps: int = 1,
    inversion_finishing_method: str = "euler",
) -> NoiseInversionOptions | None:
    """Use domain validation when inversion is selected."""
    if type(noise_inversion_enabled) is not bool:
        raise TypeError("noise_inversion_enabled must be a boolean.")
    if not noise_inversion_enabled:
        return None
    return NoiseInversionOptions(
        method=cast(InversionMethod, inversion_method),
        resolution_scale=inversion_resolution_scale,
        steps=inversion_steps,
        switch_fraction=inversion_switch_fraction,
        finishing_steps=inversion_finishing_steps,
        finishing_method=cast(InversionMethod, inversion_finishing_method),
    )
