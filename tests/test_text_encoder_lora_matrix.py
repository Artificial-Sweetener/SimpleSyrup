"""Prove the closed P9.4 text-encoder LoRA matrix."""

from __future__ import annotations

from collections import Counter

from tools.text_encoder_lora_integration.matrix import (
    TextEncoderLoraSpatialMode,
    cases,
)


def test_matrix_covers_global_regional_mixed_and_all_spatial_modes() -> None:
    """Require exact ordered references for every supported CLIP-LoRA path."""

    definitions = cases()
    ids = tuple(case.case_id for case in definitions)

    assert len(definitions) == 10
    assert len(set(ids)) == len(ids)
    assert Counter(case.spatial_mode for case in definitions) == {
        TextEncoderLoraSpatialMode.FULL: 6,
        TextEncoderLoraSpatialMode.TILED: 2,
        TextEncoderLoraSpatialMode.CONTEXTUAL: 2,
    }
    assert any(case.global_text_lora for case in definitions)
    assert any(case.regional_text_lora for case in definitions)
    assert any(case.has_text_lora and case.regional_model_lora for case in definitions)
    for index, case in enumerate(definitions):
        if case.comparison_case_id is None:
            continue
        assert case.has_text_lora
        assert case.comparison_case_id in ids[:index]
        reference = definitions[ids.index(case.comparison_case_id)]
        assert reference.spatial_mode is case.spatial_mode
        assert reference.regional_model_lora is case.regional_model_lora
