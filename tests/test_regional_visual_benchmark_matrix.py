"""Verify the complete corrected P10.3 visual matrix."""

from collections import Counter

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.regional_visual_benchmark.matrix import (
    RegionalStrategy,
    SpatialProfile,
    source_positions,
    visual_positions,
)


def test_matrix_expands_every_case_seed_profile_and_strategy() -> None:
    """Require 36 neutral sources and all 360 scored outputs."""

    manifest = load_manifest()
    sources = source_positions(manifest)
    positions = visual_positions(manifest)

    assert len(sources) == 36
    assert len({source.source_id for source in sources}) == 36
    assert len(positions) == 360
    assert len({position.artifact_id for position in positions}) == 360
    assert Counter(position.strategy for position in positions) == {
        RegionalStrategy.REGIONAL_CONDITIONING: 180,
        RegionalStrategy.ATTENTION_COUPLING: 180,
    }
    assert Counter(position.spatial_profile for position in positions) == dict.fromkeys(
        SpatialProfile, 72
    )


def test_only_corrected_spatial_positions_consume_neutral_sources() -> None:
    """Keep full generation from noise and every spatial path at 1.5x."""

    manifest = load_manifest()
    for position in visual_positions(manifest):
        assert position.is_refinement == (
            position.spatial_profile is not SpatialProfile.FULL
        )
        if position.is_refinement:
            assert position.source_id == (
                f"source__{position.case_id}__seed-{position.seed}"
            )
            assert position.expected_size == (1536, 1536)
        else:
            assert position.source_id is None
            assert position.expected_size == (1024, 1024)
