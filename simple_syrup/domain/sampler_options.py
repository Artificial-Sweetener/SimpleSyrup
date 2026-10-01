# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026 Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Assemble immutable, order-independent sampler capability configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TypeAlias

from .noise_inversion import NoiseInversionOptions
from .regional_prompting import validate_regional_prompt_weight
from .tiled_diffusion import validate_tiled_diffusion_mode


@dataclass(frozen=True, slots=True)
class TilingOptions:
    """Own the single local tile layout and blending configuration."""

    diffusion_mode: str = "multidiffusion"
    width: int = 128
    height: int = 128
    overlap: int = 32
    batch_size: int = 4
    differential_diffusion: bool = False
    segs: object | None = None

    def __post_init__(self) -> None:
        """Reject tile settings that cannot form a bounded prediction plan."""
        validate_tiled_diffusion_mode(self.diffusion_mode)
        for name, value in (("width", self.width), ("height", self.height)):
            if type(value) is not int or not 16 <= value <= 512:
                raise ValueError(
                    f"Tile {name} must be between 16 and 512 latent pixels."
                )
        if type(self.overlap) is not int or not 0 <= self.overlap < min(
            self.width, self.height
        ):
            raise ValueError(
                "Tile overlap must be nonnegative and smaller than both dimensions."
            )
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("Tile batch size must be a positive integer.")
        if type(self.differential_diffusion) is not bool:
            raise TypeError("Differential diffusion must be a boolean.")


@dataclass(frozen=True, slots=True)
class ContextualDiffusionOptions:
    """Add global scene context to the authoritative local tile predictions.

    With no Tiling options, the context size also supplies the square local
    tile size, using 32 latent pixels of overlap and a batch size of four.
    """

    context_size: int = 96
    global_weight: float = 1.0
    global_steps: int = 1
    global_decay: float = 0.5

    def __post_init__(self) -> None:
        """Reject unsupported global-context geometry and schedule values."""
        if type(self.context_size) is not int or not 16 <= self.context_size <= 512:
            raise ValueError("Context size must be between 16 and 512 latent pixels.")
        if not math.isfinite(self.global_weight) or not 0 <= self.global_weight <= 2:
            raise ValueError("Global context weight must be between 0 and 2.")
        if type(self.global_steps) is not int or self.global_steps < 0:
            raise ValueError("Global context steps must be a nonnegative integer.")
        if not math.isfinite(self.global_decay) or not 0 <= self.global_decay <= 1:
            raise ValueError("Global context decay must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class AttentionCouplingOptions:
    """Carry regional attention inputs without preparing or mutating a model."""

    region_masks: object
    regional_prompt_weight: float = 1.0
    region_mask_feather: int = 0

    def __post_init__(self) -> None:
        """Require a mask source and valid regional attention strengths."""
        if self.region_masks is None:
            raise ValueError("Attention Coupling requires region masks.")
        validate_regional_prompt_weight(self.regional_prompt_weight)
        if type(self.region_mask_feather) is not int or self.region_mask_feather < 0:
            raise ValueError("Region mask feather must be a nonnegative integer.")


SamplerCapability: TypeAlias = (
    TilingOptions
    | ContextualDiffusionOptions
    | NoiseInversionOptions
    | AttentionCouplingOptions
)


@dataclass(frozen=True, slots=True)
class SamplerOptions:
    """Own one immutable setting per capability, independent of graph order."""

    tiling: TilingOptions | None = None
    contextual_diffusion: ContextualDiffusionOptions | None = None
    noise_inversion: NoiseInversionOptions | None = None
    attention_coupling: AttentionCouplingOptions | None = None

    def __post_init__(self) -> None:
        """Reject malformed connection payloads at the typed configuration boundary."""
        for name, expected in (
            ("tiling", TilingOptions),
            ("contextual_diffusion", ContextualDiffusionOptions),
            ("noise_inversion", NoiseInversionOptions),
            ("attention_coupling", AttentionCouplingOptions),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"Sampler option {name} must be {expected.__name__}.")

    def with_capability(self, capability: SamplerCapability) -> SamplerOptions:
        """Return a fresh configuration or reject an ambiguous duplicate feature."""
        names: dict[type[object], str] = {
            TilingOptions: "tiling",
            ContextualDiffusionOptions: "contextual_diffusion",
            NoiseInversionOptions: "noise_inversion",
            AttentionCouplingOptions: "attention_coupling",
        }
        name = names.get(type(capability))
        if name is None:
            raise TypeError("Unsupported sampler capability configuration.")
        if getattr(self, name) is not None:
            raise ValueError(
                f"Duplicate sampler capability: {name}. Bypass or remove one node."
            )
        if isinstance(capability, TilingOptions):
            return replace(self, tiling=capability)
        if isinstance(capability, ContextualDiffusionOptions):
            return replace(self, contextual_diffusion=capability)
        if isinstance(capability, NoiseInversionOptions):
            return replace(self, noise_inversion=capability)
        return replace(self, attention_coupling=capability)


def append_sampler_capability(
    options: SamplerOptions | None, capability: SamplerCapability
) -> SamplerOptions:
    """Start an options chain or append to a validated incoming connection."""
    if options is not None and not isinstance(options, SamplerOptions):
        raise TypeError(
            "Options input must be a SimpleSyrup sampler options connection."
        )
    return (options if options is not None else SamplerOptions()).with_capability(
        capability
    )
