"""Verify the closed P10.2 strategy comparison matrix."""

from tools.regional_strategy_comparison.matrix import (
    SOURCE_CASE_ID,
    StrategyComparisonCase,
    cases,
    expected_low_rank_multiplier,
)


def test_matrix_covers_matched_modes_and_regional_lora_scaling() -> None:
    """Keep both strategies matched and LoRA scaling on each major path."""

    definitions = cases()
    assert len(definitions) == 16
    assert definitions[0].case_id == SOURCE_CASE_ID
    assert len({case.case_id for case in definitions}) == len(definitions)

    by_profile: dict[str, list[StrategyComparisonCase]] = {}
    for case in definitions:
        by_profile.setdefault(case.spatial_profile, []).append(case)
    assert {case.strategy for case in by_profile["full"]} == {
        "regional_conditioning",
        "attention_coupling",
    }
    for profile in ("full", "tiled_multidiffusion", "contextual_multidiffusion"):
        attention = [
            case
            for case in by_profile[profile]
            if case.strategy == "attention_coupling"
        ]
        assert [case.regional_lora_count for case in attention] == [0, 1, 4]
        assert [expected_low_rank_multiplier(case) for case in attention] == [
            0.0,
            2.0,
            8.0,
        ]


def test_only_attention_coupling_cases_declare_regional_loras() -> None:
    """Keep model-side regional adapters out of the legacy control cases."""

    assert all(
        case.strategy == "attention_coupling"
        for case in cases()
        if case.regional_lora_count
    )
