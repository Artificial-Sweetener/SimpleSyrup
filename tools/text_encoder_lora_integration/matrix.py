# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the immutable P9.4 text-encoder LoRA evidence matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

MASK_CASE_ID = "vertical-hard-50-50"


class TextEncoderLoraSpatialMode(StrEnum):
    """Identify one public Attention Coupling spatial execution path."""

    FULL = "full"
    TILED = "tiled"
    CONTEXTUAL = "contextual"


@dataclass(frozen=True, slots=True)
class TextEncoderLoraCase:
    """Describe one exact global/regional CLIP-LoRA workflow."""

    case_id: str
    label: str
    spatial_mode: TextEncoderLoraSpatialMode
    cfg: float
    global_text_lora: bool = False
    regional_text_lora: bool = False
    regional_model_lora: bool = False
    comparison_case_id: str | None = None
    mask_case_id: str = MASK_CASE_ID
    feather: int = 0

    def __post_init__(self) -> None:
        """Validate stable artifact and comparison relationships."""

        if not self.case_id or not self.label:
            raise ValueError("P9.4 case identity and label must be non-empty.")
        if not isinstance(self.spatial_mode, TextEncoderLoraSpatialMode):
            raise TypeError("P9.4 spatial mode has an invalid type.")
        if self.cfg <= 0.0:
            raise ValueError("P9.4 CFG must be positive.")
        if self.comparison_case_id == self.case_id:
            raise ValueError("P9.4 case cannot compare with itself.")

    @property
    def has_text_lora(self) -> bool:
        """Return whether this case must prove a CLIP-LoRA image effect."""

        return self.global_text_lora or self.regional_text_lora


def cases() -> tuple[TextEncoderLoraCase, ...]:
    """Return ordered full, tiled, and Contextual comparison pairs."""

    full = TextEncoderLoraSpatialMode.FULL
    tiled = TextEncoderLoraSpatialMode.TILED
    contextual = TextEncoderLoraSpatialMode.CONTEXTUAL
    return (
        TextEncoderLoraCase(
            "full-baseline",
            "Full-context baseline without LoRA",
            full,
            4.0,
        ),
        TextEncoderLoraCase(
            "full-global-text_adapter-text",
            "Full-context global compatible TEXT_ADAPTER Qwen text-encoder LoRA",
            full,
            4.0,
            global_text_lora=True,
            comparison_case_id="full-baseline",
        ),
        TextEncoderLoraCase(
            "full-regional-text_adapter-text",
            "Full-context left-region compatible TEXT_ADAPTER Qwen text-encoder LoRA",
            full,
            4.0,
            regional_text_lora=True,
            comparison_case_id="full-baseline",
        ),
        TextEncoderLoraCase(
            "full-regional-adapter_a-model",
            "Full-context left-region ADAPTER_A model LoRA reference",
            full,
            4.0,
            regional_model_lora=True,
        ),
        TextEncoderLoraCase(
            "full-regional-text_adapter-text-adapter_a-model",
            "Full-context left-region compatible TEXT_ADAPTER text plus ADAPTER_A model LoRAs",
            full,
            4.0,
            regional_text_lora=True,
            regional_model_lora=True,
            comparison_case_id="full-regional-adapter_a-model",
        ),
        TextEncoderLoraCase(
            "full-global-text_adapter-text-regional-adapter_a-model",
            "Full-context global compatible TEXT_ADAPTER text plus regional "
            "ADAPTER_A model LoRA",
            full,
            4.0,
            global_text_lora=True,
            regional_model_lora=True,
            comparison_case_id="full-regional-adapter_a-model",
        ),
        TextEncoderLoraCase(
            "tiled-regional-adapter_a-model",
            "Tiled 1.5x refinement with regional ADAPTER_A model LoRA reference",
            tiled,
            1.0,
            regional_model_lora=True,
        ),
        TextEncoderLoraCase(
            "tiled-regional-text_adapter-text-adapter_a-model",
            "Tiled 1.5x refinement with compatible TEXT_ADAPTER text and ADAPTER_A model LoRAs",
            tiled,
            1.0,
            regional_text_lora=True,
            regional_model_lora=True,
            comparison_case_id="tiled-regional-adapter_a-model",
        ),
        TextEncoderLoraCase(
            "contextual-regional-adapter_a-model",
            "Contextual 1.5x refinement with regional ADAPTER_A model LoRA reference",
            contextual,
            1.0,
            regional_model_lora=True,
        ),
        TextEncoderLoraCase(
            "contextual-regional-text_adapter-text-adapter_a-model",
            "Contextual 1.5x refinement with compatible TEXT_ADAPTER text and "
            "ADAPTER_A model LoRAs",
            contextual,
            1.0,
            regional_text_lora=True,
            regional_model_lora=True,
            comparison_case_id="contextual-regional-adapter_a-model",
        ),
    )
