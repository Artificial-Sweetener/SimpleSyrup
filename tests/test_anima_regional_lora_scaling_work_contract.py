"""Verify one shared declared-to-observed scaling work contract."""

from __future__ import annotations

from tools.anima_regional_lora_performance.matrix_manifest import (
    default_scaling_manifest,
)
from tools.anima_regional_lora_performance.scaling_work_contract import (
    expected_scaling_work,
)


def test_expected_scaling_work_preserves_every_declared_cardinality() -> None:
    """Map every work field without inferring runtime policy."""

    expectation = default_scaling_manifest().profiles[9].work

    observed = expected_scaling_work(expectation)

    assert observed.active_adapter_uses == expectation.active_adapter_uses
    assert observed.active_target_count == expectation.active_target_count
    assert observed.target_use_count == expectation.target_use_count
    assert (
        observed.deduplicated_target_group_count
        == expectation.deduplicated_target_group_count
    )
    assert (
        observed.compatible_projection_batch_count
        == expectation.compatible_projection_batch_count
    )
    assert observed.deduplicated_target_uses == expectation.deduplicated_target_uses
