"""Validate decoded-pixel equality for exact P9.7 full-context paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.decoded_image_comparison import compare_decoded_rgb_images

_BASELINE_CASE_ID = "anima-full-baseline"
_EXACT_CASE_IDS = (
    "anima-full-easycache",
    "anima-full-optimized-attention",
)


@dataclass(frozen=True, slots=True)
class ExactImageEquivalence:
    """Record one decoded-pixel-identical optimized path."""

    baseline_case_id: str
    compared_case_id: str
    changed_pixels: int
    maximum_channel_delta: int
    width: int
    height: int


def validate_exact_image_equivalence(
    accepted_images: dict[str, Path],
) -> tuple[ExactImageEquivalence, ...]:
    """Require admitted exact full-context modifiers to match baseline pixels."""

    compared_ids = tuple(
        case_id for case_id in _EXACT_CASE_IDS if case_id in accepted_images
    )
    if not compared_ids:
        return ()
    baseline = accepted_images.get(_BASELINE_CASE_ID)
    if baseline is None:
        raise ValueError("P9.7 exact image comparison requires the full baseline.")
    evidence: list[ExactImageEquivalence] = []
    for case_id in compared_ids:
        comparison = compare_decoded_rgb_images(baseline, accepted_images[case_id])
        if comparison.changed_pixels or comparison.maximum_channel_delta:
            raise ValueError(
                f"P9.7 exact path {case_id} changed decoded baseline pixels: "
                f"changed_pixels={comparison.changed_pixels}, "
                f"maximum_channel_delta={comparison.maximum_channel_delta}."
            )
        evidence.append(
            ExactImageEquivalence(
                _BASELINE_CASE_ID,
                case_id,
                comparison.changed_pixels,
                comparison.maximum_channel_delta,
                comparison.width,
                comparison.height,
            )
        )
    return tuple(evidence)
