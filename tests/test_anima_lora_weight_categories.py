"""Verify complete authoritative Anima multiplier category classification."""

from __future__ import annotations

from collections import Counter

from simple_syrup.runtime.regional_lora.anima_lora_weight_categories import (
    AnimaLoraWeightCategory,
    anima_lora_weight_category,
)
from simple_syrup.runtime.regional_lora.anima_targets import AnimaLoraTargetFamily


def test_weight_categories_cover_every_target_family_once() -> None:
    """Pin all sixteen admitted families to their one multiplier scope."""

    counts = Counter(
        anima_lora_weight_category(family) for family in AnimaLoraTargetFamily
    )

    assert counts == {
        AnimaLoraWeightCategory.SELF_ATTENTION: 4,
        AnimaLoraWeightCategory.CROSS_ATTENTION: 4,
        AnimaLoraWeightCategory.MLP: 2,
        AnimaLoraWeightCategory.ADALN: 6,
    }
