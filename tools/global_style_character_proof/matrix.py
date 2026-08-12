"""Define fixed CHARACTER_A-region and global-ADAPTER_A comparison cases."""

from __future__ import annotations

from dataclasses import dataclass

ADAPTER_A_NAME = r"Anima\style\adapter-a.safetensors"


@dataclass(frozen=True, slots=True)
class GlobalStyleCharacterCase:
    """Describe one whole-image style strength with right-region CHARACTER_A."""

    case_id: str
    label: str
    global_style_strength: float | None


def cases() -> tuple[GlobalStyleCharacterCase, ...]:
    """Return the character control and two whole-image style strengths."""

    return (
        GlobalStyleCharacterCase(
            "character_a-right-only",
            "character_a RIGHT 1.0 — no global style",
            None,
        ),
        GlobalStyleCharacterCase(
            "global-adapter_a-050-character_a-right",
            "GLOBAL ADAPTER_A 0.5 + character_a RIGHT 1.0",
            0.5,
        ),
        GlobalStyleCharacterCase(
            "global-adapter_a-100-character_a-right",
            "GLOBAL ADAPTER_A 1.0 + character_a RIGHT 1.0",
            1.0,
        ),
    )
