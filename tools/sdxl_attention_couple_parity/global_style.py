# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the optional generic global-style control for backend parity."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tools.sdxl_attention_coupling_integration.graph import (
    NodeReference,
    SdxlWorkflowGraph,
)


@dataclass(frozen=True, slots=True)
class ParityGlobalStyle:
    """Declare one ordinary MODEL+CLIP adapter and base-only trigger pair."""

    lora_name: str
    strength: float
    prompt_g: str
    prompt_l: str

    def __post_init__(self) -> None:
        """Reject incomplete or non-finite parity controls."""

        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.lora_name, self.prompt_g, self.prompt_l)
        ):
            raise ValueError("Parity global-style values must be non-empty strings.")
        if isinstance(self.strength, bool) or not isinstance(
            self.strength, int | float
        ):
            raise TypeError("Parity global-style strength must be a real number.")
        if not math.isfinite(float(self.strength)):
            raise ValueError("Parity global-style strength must be finite.")


@dataclass(frozen=True, slots=True)
class LoadedParityGlobalStyle:
    """Expose the equally patched model and text encoders."""

    model: NodeReference
    clip: NodeReference


def load_parity_global_style(
    graph: SdxlWorkflowGraph,
    *,
    style: ParityGlobalStyle,
    model: NodeReference,
    clip: NodeReference,
) -> LoadedParityGlobalStyle:
    """Apply one ordinary global adapter before either parity backend."""

    if not isinstance(graph, SdxlWorkflowGraph):
        raise TypeError("Parity global style requires an SDXL workflow graph.")
    if not isinstance(style, ParityGlobalStyle):
        raise TypeError("Parity global style has an invalid declaration.")
    loaded = graph.add(
        "LoraLoader",
        model=model,
        clip=clip,
        lora_name=style.lora_name,
        strength_model=style.strength,
        strength_clip=style.strength,
    )
    return LoadedParityGlobalStyle([loaded, 0], [loaded, 1])


def append_base_style(prompt: str, addition: str) -> str:
    """Append one trigger only to its global/base prompt."""

    if not prompt.strip() or not addition.strip():
        raise ValueError("Parity base style prompts must be non-empty.")
    return f"{prompt}, {addition}"
