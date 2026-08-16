# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare paired global and regional full-strength character-LoRA cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FidelityExecutionMode(StrEnum):
    """Identify ordinary global or all-one regional LoRA placement."""

    GLOBAL_REFERENCE = "global-reference"
    REGIONAL_ALL_ONE = "regional-all-one"


@dataclass(frozen=True, slots=True)
class CharacterFidelitySpec:
    """Describe one externally supplied character adapter and trigger prompt."""

    character_id: str
    label: str
    adapter_name: str
    prompt_g: str
    prompt_l: str

    def __post_init__(self) -> None:
        """Require complete generic external case data."""

        values = (
            self.character_id,
            self.label,
            self.adapter_name,
            self.prompt_g,
            self.prompt_l,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Character fidelity values must be non-empty strings.")


@dataclass(frozen=True, slots=True)
class FullStrengthFidelityCase:
    """Retain one exact full-strength placement comparison case."""

    case_id: str
    label: str
    mode: FidelityExecutionMode
    character: CharacterFidelitySpec
    model_strength: float = 1.0
    clip_strength: float = 1.0
    schedule: tuple[tuple[float, float], ...] = ((0.0, 1.0),)


def fidelity_cases(
    characters: tuple[CharacterFidelitySpec, ...],
) -> tuple[FullStrengthFidelityCase, ...]:
    """Return adjacent global/all-one-regional pairs for every character."""

    if not characters:
        raise ValueError("Full-strength fidelity requires at least one character.")
    if len({character.character_id for character in characters}) != len(characters):
        raise ValueError("Character fidelity IDs must be unique.")
    return tuple(
        case
        for character in characters
        for case in (
            FullStrengthFidelityCase(
                f"{character.character_id}-global-reference",
                f"{character.label} — ordinary global LoRA 1.0 reference",
                FidelityExecutionMode.GLOBAL_REFERENCE,
                character,
            ),
            FullStrengthFidelityCase(
                f"{character.character_id}-regional-all-one",
                f"{character.label} — regional LoRA 1.0, all-one mask",
                FidelityExecutionMode.REGIONAL_ALL_ONE,
                character,
            ),
        )
    )
